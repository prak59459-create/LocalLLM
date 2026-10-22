import Foundation

/// Jinja テンプレート（Hugging Face のチャットテンプレート用のサブセット）。
/// transformers の apply_chat_template と同じ設定（trim_blocks / lstrip_blocks、loopcontrols、tojson など）で描画する。
struct JinjaTemplate {
    private let nodes: [JinjaNode]

    init(_ source: String, trimBlocks: Bool = true, lstripBlocks: Bool = true) throws {
        var lexer = JinjaLexer(source, trimBlocks: trimBlocks, lstripBlocks: lstripBlocks)
        var parser = JinjaParser(segments: try lexer.tokenize())
        nodes = try parser.parse()
    }

    func render(_ context: [String: JinjaValue]) throws -> String {
        let interpreter = JinjaInterpreter(globals: context)
        var output = ""
        _ = try interpreter.execute(nodes, into: &output)
        return output
    }
}

/// テンプレートを実行する。
final class JinjaInterpreter {
    enum Flow {
        case normal
        case breakLoop
        case continueLoop
    }

    /// 変数のスコープ（for の 1 回ごと、マクロの呼び出しごとに 1 つ増える）
    private var scopes: [[String: JinjaValue]]

    init(globals: [String: JinjaValue]) {
        var root = JinjaBuiltins.globals
        for (key, value) in globals { root[key] = value }
        scopes = [root]
    }

    // MARK: - 文

    func execute(_ nodes: [JinjaNode], into output: inout String) throws -> Flow {
        for node in nodes {
            let flow = try execute(node, into: &output)
            if flow != .normal { return flow }
        }
        return .normal
    }

    private func execute(_ node: JinjaNode, into output: inout String) throws -> Flow {
        switch node {
        case .text(let text):
            output += text
        case .output(let expr):
            output += try evaluate(expr).renderedString
        case .ifBlock(let branches, let elseBody):
            for (condition, body) in branches {
                if try evaluate(condition).truthy {
                    return try execute(body, into: &output)
                }
            }
            if let elseBody { return try execute(elseBody, into: &output) }
        case .forBlock(let target, let iterableExpr, let filter, let body, let elseBody):
            return try executeFor(target, iterableExpr, filter, body, elseBody, into: &output)
        case .set(let target, let expr):
            try assign(target, try evaluate(expr))
        case .setBlock(let name, let body):
            var captured = ""
            _ = try execute(body, into: &captured)
            scopes[scopes.count - 1][name] = .string(captured)
        case .macro(let name, let params, let body):
            defineMacro(name, params, body)
        case .filterBlock(let name, let args, let kwargs, let body):
            var captured = ""
            _ = try execute(body, into: &captured)
            let arguments = try args.map { try evaluate($0) }
            let keywords = try kwargs.map { ($0.0, try evaluate($0.1)) }
            output += try JinjaBuiltins.applyFilter(name, .string(captured), arguments, keywords, interpreter: self).renderedString
        case .breakLoop:
            return .breakLoop
        case .continueLoop:
            return .continueLoop
        }
        return .normal
    }

    private func executeFor(_ target: JinjaTarget, _ iterableExpr: JinjaExpr, _ filter: JinjaExpr?, _ body: [JinjaNode],
                            _ elseBody: [JinjaNode]?, into output: inout String) throws -> Flow {
        let iterable = try evaluate(iterableExpr)
        guard var items = iterable.iterationItems else {
            throw JinjaError.runtime("'\(iterable.typeName)' object is not iterable")
        }
        // for ... if ... の絞り込み（loop の値は絞り込み後で数える）
        if let filter {
            var kept: [JinjaValue] = []
            for item in items {
                scopes.append([:])
                defer { scopes.removeLast() }
                try bind(target, item)
                if try evaluate(filter).truthy { kept.append(item) }
            }
            items = kept
        }
        if items.isEmpty {
            if let elseBody { return try execute(elseBody, into: &output) }
            return .normal
        }
        let count = items.count
        for (index, item) in items.enumerated() {
            var loop = JinjaObject()
            loop["index"] = .int(index + 1)
            loop["index0"] = .int(index)
            loop["revindex"] = .int(count - index)
            loop["revindex0"] = .int(count - index - 1)
            loop["first"] = .bool(index == 0)
            loop["last"] = .bool(index == count - 1)
            loop["length"] = .int(count)
            loop["depth"] = .int(1)
            loop["depth0"] = .int(0)
            if index > 0 { loop["previtem"] = items[index - 1] }
            if index < count - 1 { loop["nextitem"] = items[index + 1] }
            loop["cycle"] = .function(JinjaFunction(name: "cycle") { args, _ in
                args.isEmpty ? .undefined : args[index % args.count]
            })
            scopes.append(["loop": .object(loop)])
            defer { scopes.removeLast() }
            try bind(target, item)
            let flow = try execute(body, into: &output)
            if flow == .breakLoop { break }
        }
        return .normal
    }

    /// for の変数に値を入れる（現在のスコープ）。
    private func bind(_ target: JinjaTarget, _ value: JinjaValue) throws {
        switch target {
        case .name(let name):
            scopes[scopes.count - 1][name] = value
        case .tuple(let names):
            guard let items = value.iterationItems, items.count == names.count else {
                throw JinjaError.runtime("cannot unpack \(value.typeName) into \(names.count) values")
            }
            for (name, item) in zip(names, items) { scopes[scopes.count - 1][name] = item }
        case .attribute:
            throw JinjaError.runtime("cannot use an attribute as a loop variable")
        }
    }

    private func assign(_ target: JinjaTarget, _ value: JinjaValue) throws {
        switch target {
        case .name, .tuple:
            try bind(target, value)
        case .attribute(let name, let attribute):
            guard case .namespace(let namespace) = lookup(name) else {
                throw JinjaError.runtime("cannot assign attribute on non-namespace object '\(name)'")
            }
            namespace.attributes[attribute] = value
        }
    }

    private func defineMacro(_ name: String, _ params: [(String, JinjaExpr?)], _ body: [JinjaNode]) {
        let function = JinjaFunction(name: name) { [unowned self] args, kwargs in
            var locals: [String: JinjaValue] = [:]
            for (index, param) in params.enumerated() {
                if index < args.count {
                    locals[param.0] = args[index]
                } else if let value = kwargs.first(where: { $0.0 == param.0 })?.1 {
                    locals[param.0] = value
                } else if let defaultExpr = param.1 {
                    locals[param.0] = try self.evaluate(defaultExpr)
                } else {
                    locals[param.0] = .undefined
                }
            }
            locals["varargs"] = .array(Array(args.dropFirst(params.count)))
            let saved = self.scopes
            // マクロはテンプレート全体の変数と引数だけを見る
            self.scopes = [self.scopes[0], locals]
            defer { self.scopes = saved }
            var output = ""
            _ = try self.execute(body, into: &output)
            return .string(output)
        }
        scopes[scopes.count - 1][name] = .function(function)
    }

    func lookup(_ name: String) -> JinjaValue {
        for scope in scopes.reversed() {
            if let value = scope[name] { return value }
        }
        return .undefined
    }

    // MARK: - 式

    func evaluate(_ expr: JinjaExpr) throws -> JinjaValue {
        switch expr {
        case .literal(let value):
            return value
        case .name(let name):
            return lookup(name)
        case .list(let items):
            return .array(try items.map { try evaluate($0) })
        case .tuple(let items):
            return .tuple(try items.map { try evaluate($0) })
        case .dict(let pairs):
            var object = JinjaObject()
            for (keyExpr, valueExpr) in pairs {
                let key = try evaluate(keyExpr)
                object[key.pythonStr] = try evaluate(valueExpr)
            }
            return .object(object)
        case .attribute(let baseExpr, let name):
            let base = try evaluate(baseExpr)
            return try attribute(of: base, name, describing: baseExpr)
        case .item(let baseExpr, let keyExpr):
            let base = try evaluate(baseExpr)
            if base.isUndefined { throw undefinedError(baseExpr) }
            return try JinjaBuiltins.subscriptValue(base, try evaluate(keyExpr))
        case .slice(let baseExpr, let startExpr, let stopExpr, let stepExpr):
            let base = try evaluate(baseExpr)
            if base.isUndefined { throw undefinedError(baseExpr) }
            return try JinjaBuiltins.slice(base,
                                           start: try startExpr.map { try evaluate($0) },
                                           stop: try stopExpr.map { try evaluate($0) },
                                           step: try stepExpr.map { try evaluate($0) })
        case .call(let calleeExpr, let argExprs, let kwargExprs):
            let args = try argExprs.map { try evaluate($0) }
            let kwargs = try kwargExprs.map { ($0.0, try evaluate($0.1)) }
            // メソッド呼び出し（"abc".strip() など）
            if case .attribute(let baseExpr, let name) = calleeExpr {
                let base = try evaluate(baseExpr)
                if base.isUndefined { throw undefinedError(baseExpr) }
                if case .object(let object) = base, case .function(let function)? = object[name] {
                    return try function.body(args, kwargs)
                }
                if case .namespace(let namespace) = base, case .function(let function)? = namespace.attributes[name] {
                    return try function.body(args, kwargs)
                }
                if let result = try JinjaBuiltins.callMethod(base, name, args, kwargs) {
                    return result
                }
                throw JinjaError.runtime("'\(base.typeName)' object has no attribute '\(name)'")
            }
            let callee = try evaluate(calleeExpr)
            guard case .function(let function) = callee else {
                if callee.isUndefined { throw undefinedError(calleeExpr) }
                throw JinjaError.runtime("'\(callee.typeName)' object is not callable")
            }
            return try function.body(args, kwargs)
        case .filter(let baseExpr, let name, let argExprs, let kwargExprs):
            let base = try evaluate(baseExpr)
            let args = try argExprs.map { try evaluate($0) }
            let kwargs = try kwargExprs.map { ($0.0, try evaluate($0.1)) }
            return try JinjaBuiltins.applyFilter(name, base, args, kwargs, interpreter: self)
        case .test(let baseExpr, let name, let argExprs, let negated):
            let base = try evaluate(baseExpr)
            let args = try argExprs.map { try evaluate($0) }
            let result = try JinjaBuiltins.applyTest(name, base, args, interpreter: self)
            return .bool(negated ? !result : result)
        case .not(let inner):
            return .bool(!(try evaluate(inner).truthy))
        case .negate(let inner):
            switch try evaluate(inner) {
            case .int(let value): return .int(-value)
            case .double(let value): return .double(-value)
            case .bool(let value): return .int(value ? -1 : 0)
            case let other: throw JinjaError.runtime("bad operand type for unary -: '\(other.typeName)'")
            }
        case .plus(let inner):
            return try evaluate(inner)
        case .and(let left, let right):
            let value = try evaluate(left)
            return value.truthy ? try evaluate(right) : value
        case .or(let left, let right):
            let value = try evaluate(left)
            return value.truthy ? value : try evaluate(right)
        case .binary(let op, let left, let right):
            return try JinjaBuiltins.binary(op, try evaluate(left), try evaluate(right))
        case .compare(let firstExpr, let rest):
            var left = try evaluate(firstExpr)
            for (op, rightExpr) in rest {
                let right = try evaluate(rightExpr)
                guard try JinjaBuiltins.compare(op, left, right) else { return .bool(false) }
                left = right
            }
            return .bool(true)
        case .conditional(let condition, let then, let otherwise):
            if try evaluate(condition).truthy { return try evaluate(then) }
            return try otherwise.map { try evaluate($0) } ?? .undefined
        }
    }

    private func attribute(of base: JinjaValue, _ name: String, describing expr: JinjaExpr) throws -> JinjaValue {
        switch base {
        case .undefined:
            throw undefinedError(expr)
        case .object(let object):
            if let value = object[name] { return value }
            if let method = JinjaBuiltins.boundMethod(base, name) { return method }
            return .undefined
        case .namespace(let namespace):
            return namespace.attributes[name] ?? .undefined
        default:
            if let method = JinjaBuiltins.boundMethod(base, name) { return method }
            return .undefined
        }
    }

    private func undefinedError(_ expr: JinjaExpr) -> JinjaError {
        if case .name(let name) = expr { return .runtime("'\(name)' is undefined") }
        return .runtime("attribute of an undefined value")
    }
}
