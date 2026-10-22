import Foundation

/// Jinja の組み込み関数・フィルタ・テスト・メソッド。
enum JinjaBuiltins {
    // MARK: - グローバル関数

    static var globals: [String: JinjaValue] {
        [
            "range": .function(JinjaFunction(name: "range") { args, _ in
                let numbers = args.compactMap { $0.numericValue }.map { Int($0) }
                var start = 0, stop = 0, step = 1
                switch numbers.count {
                case 1: stop = numbers[0]
                case 2: start = numbers[0]; stop = numbers[1]
                case 3: start = numbers[0]; stop = numbers[1]; step = numbers[2]
                default: throw JinjaError.runtime("range expected 1 to 3 arguments")
                }
                guard step != 0 else { throw JinjaError.runtime("range() arg 3 must not be zero") }
                var values: [JinjaValue] = []
                var current = start
                while step > 0 ? current < stop : current > stop {
                    values.append(.int(current))
                    current += step
                    if values.count > 1_000_000 { break }
                }
                return .array(values)
            }),
            "namespace": .function(JinjaFunction(name: "namespace") { args, kwargs in
                var object = JinjaObject()
                if case .object(let initial)? = args.first { object = initial }
                for (key, value) in kwargs { object[key] = value }
                return .namespace(JinjaNamespace(object))
            }),
            "dict": .function(JinjaFunction(name: "dict") { args, kwargs in
                var object = JinjaObject()
                if case .object(let initial)? = args.first { object = initial }
                for (key, value) in kwargs { object[key] = value }
                return .object(object)
            }),
            "raise_exception": .function(JinjaFunction(name: "raise_exception") { args, _ in
                throw JinjaError.raised(args.first?.renderedString ?? "")
            }),
            "strftime_now": .function(JinjaFunction(name: "strftime_now") { args, _ in
                .string(strftime(args.first?.renderedString ?? "%Y-%m-%d", Date()))
            }),
        ]
    }

    /// Python の strftime の主な書式
    static func strftime(_ format: String, _ date: Date) -> String {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone.current
        let c = calendar.dateComponents([.year, .month, .day, .hour, .minute, .second, .weekday], from: date)
        let months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
        let weekdays = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        let month = (c.month ?? 1) - 1
        let weekday = (c.weekday ?? 1) - 1
        let hour = c.hour ?? 0
        func two(_ value: Int) -> String { String(format: "%02d", value) }
        var result = ""
        var iterator = Array(format.unicodeScalars)[...]
        while let scalar = iterator.first {
            iterator = iterator.dropFirst()
            guard scalar == "%", let code = iterator.first else {
                result.unicodeScalars.append(scalar)
                continue
            }
            iterator = iterator.dropFirst()
            switch code {
            case "d": result += two(c.day ?? 1)
            case "m": result += two(c.month ?? 1)
            case "Y": result += String(c.year ?? 1970)
            case "y": result += two((c.year ?? 1970) % 100)
            case "b": result += String(months[month].prefix(3))
            case "B": result += months[month]
            case "a": result += String(weekdays[weekday].prefix(3))
            case "A": result += weekdays[weekday]
            case "H": result += two(hour)
            case "I": result += two(hour % 12 == 0 ? 12 : hour % 12)
            case "p": result += hour < 12 ? "AM" : "PM"
            case "M": result += two(c.minute ?? 0)
            case "S": result += two(c.second ?? 0)
            case "j": result += String(format: "%03d", calendar.ordinality(of: .day, in: .year, for: date) ?? 1)
            case "%": result += "%"
            default:
                result += "%"
                result.unicodeScalars.append(code)
            }
        }
        return result
    }

    // MARK: - 添字

    static func subscriptValue(_ base: JinjaValue, _ key: JinjaValue) throws -> JinjaValue {
        switch base {
        case .array(let items), .tuple(let items):
            guard let number = key.numericValue else { return .undefined }
            var index = Int(number)
            if index < 0 { index += items.count }
            return items.indices.contains(index) ? items[index] : .undefined
        case .string(let text):
            guard let number = key.numericValue else { return .undefined }
            let scalars = Array(text.unicodeScalars)
            var index = Int(number)
            if index < 0 { index += scalars.count }
            return scalars.indices.contains(index) ? .string(String(scalars[index])) : .undefined
        case .object(let object):
            return object[key.pythonStr] ?? .undefined
        case .namespace(let namespace):
            return namespace.attributes[key.pythonStr] ?? .undefined
        default:
            return .undefined
        }
    }

    /// Python のスライス [start:stop:step]
    static func slice(_ base: JinjaValue, start: JinjaValue?, stop: JinjaValue?, step: JinjaValue?) throws -> JinjaValue {
        func indices(_ count: Int) throws -> [Int] {
            let stepValue = step.flatMap { $0.numericValue }.map { Int($0) } ?? 1
            guard stepValue != 0 else { throw JinjaError.runtime("slice step cannot be zero") }
            func clamp(_ value: JinjaValue?, _ defaultValue: Int) -> Int {
                guard let number = value?.numericValue else { return defaultValue }
                var index = Int(number)
                if index < 0 { index += count }
                if stepValue > 0 { return min(max(index, 0), count) }
                return min(max(index, -1), count - 1)
            }
            var result: [Int] = []
            if stepValue > 0 {
                var index = clamp(start, 0)
                let end = clamp(stop, count)
                while index < end { result.append(index); index += stepValue }
            } else {
                var index = clamp(start, count - 1)
                let end = clamp(stop, -1)
                while index > end { result.append(index); index += stepValue }
            }
            return result
        }
        switch base {
        case .array(let items):
            return .array(try indices(items.count).map { items[$0] })
        case .tuple(let items):
            return .tuple(try indices(items.count).map { items[$0] })
        case .string(let text):
            let scalars = Array(text.unicodeScalars)
            var view = String.UnicodeScalarView()
            for index in try indices(scalars.count) { view.append(scalars[index]) }
            return .string(String(view))
        default:
            throw JinjaError.runtime("'\(base.typeName)' object is not subscriptable")
        }
    }

    // MARK: - 演算

    static func binary(_ op: String, _ left: JinjaValue, _ right: JinjaValue) throws -> JinjaValue {
        if op == "~" { return .string(left.renderedString + right.renderedString) }
        switch (op, left, right) {
        case ("+", .string(let a), .string(let b)): return .string(a + b)
        case ("+", .array(let a), .array(let b)): return .array(a + b)
        case ("+", .tuple(let a), .tuple(let b)): return .tuple(a + b)
        case ("*", .string(let a), .int(let b)): return .string(String(repeating: a, count: max(b, 0)))
        case ("*", .int(let a), .string(let b)): return .string(String(repeating: b, count: max(a, 0)))
        case ("*", .array(let a), .int(let b)): return .array(Array(repeating: a, count: max(b, 0)).flatMap { $0 })
        case ("%", .string(let format), _): return .string(try percentFormat(format, right))
        default: break
        }
        let intPair = integerPair(left, right)
        guard let a = left.numericValue, let b = right.numericValue else {
            throw JinjaError.runtime("unsupported operand type(s) for \(op): '\(left.typeName)' and '\(right.typeName)'")
        }
        switch op {
        case "+":
            if let (x, y) = intPair { return .int(x &+ y) }
            return .double(a + b)
        case "-":
            if let (x, y) = intPair { return .int(x &- y) }
            return .double(a - b)
        case "*":
            if let (x, y) = intPair { return .int(x &* y) }
            return .double(a * b)
        case "/":
            guard b != 0 else { throw JinjaError.runtime("division by zero") }
            return .double(a / b)
        case "//":
            guard b != 0 else { throw JinjaError.runtime("integer division or modulo by zero") }
            if let (x, y) = intPair {
                let quotient = x / y
                return .int((x % y != 0 && (x < 0) != (y < 0)) ? quotient - 1 : quotient)
            }
            return .double((a / b).rounded(.down))
        case "%":
            guard b != 0 else { throw JinjaError.runtime("integer division or modulo by zero") }
            if let (x, y) = intPair {
                let remainder = x % y
                return .int(remainder != 0 && (remainder < 0) != (y < 0) ? remainder + y : remainder)
            }
            let remainder = a.truncatingRemainder(dividingBy: b)
            return .double(remainder != 0 && (remainder < 0) != (b < 0) ? remainder + b : remainder)
        case "**":
            if let (x, y) = intPair, y >= 0 {
                var result = 1
                for _ in 0..<y { result = result &* x }
                return .int(result)
            }
            return .double(pow(a, b))
        default:
            throw JinjaError.runtime("unknown operator \(op)")
        }
    }

    private static func integerPair(_ left: JinjaValue, _ right: JinjaValue) -> (Int, Int)? {
        func int(_ value: JinjaValue) -> Int? {
            switch value {
            case .int(let number): return number
            case .bool(let flag): return flag ? 1 : 0
            default: return nil
            }
        }
        guard let a = int(left), let b = int(right) else { return nil }
        return (a, b)
    }

    /// Python の "%s" % value（%s と %d だけ）
    private static func percentFormat(_ format: String, _ value: JinjaValue) throws -> String {
        var values: [JinjaValue]
        if case .tuple(let items) = value { values = items } else { values = [value] }
        var result = ""
        var iterator = Array(format.unicodeScalars)[...]
        while let scalar = iterator.first {
            iterator = iterator.dropFirst()
            guard scalar == "%", let code = iterator.first else {
                result.unicodeScalars.append(scalar)
                continue
            }
            iterator = iterator.dropFirst()
            if code == "%" { result += "%"; continue }
            guard !values.isEmpty else { throw JinjaError.runtime("not enough arguments for format string") }
            let item = values.removeFirst()
            switch code {
            case "d", "i": result += String(Int(item.numericValue ?? 0))
            case "r": result += item.pythonRepr
            default: result += item.pythonStr
            }
        }
        return result
    }

    static func compare(_ op: String, _ left: JinjaValue, _ right: JinjaValue) throws -> Bool {
        switch op {
        case "==": return left.isEqual(to: right)
        case "!=": return !left.isEqual(to: right)
        case "<": return try left.compare(right) == .orderedAscending
        case ">": return try left.compare(right) == .orderedDescending
        case "<=": return try left.compare(right) != .orderedDescending
        case ">=": return try left.compare(right) != .orderedAscending
        case "in": return try contains(right, left)
        case "notin": return !(try contains(right, left))
        default: throw JinjaError.runtime("unknown comparison \(op)")
        }
    }

    static func contains(_ container: JinjaValue, _ item: JinjaValue) throws -> Bool {
        switch container {
        case .string(let text):
            guard case .string(let needle) = item else {
                throw JinjaError.runtime("'in <string>' requires string as left operand, not \(item.typeName)")
            }
            if needle.isEmpty { return true }
            return (text as NSString).range(of: needle, options: .literal).location != NSNotFound
        case .array(let items), .tuple(let items):
            return items.contains { $0.isEqual(to: item) }
        case .object(let object):
            return object[item.pythonStr] != nil
        case .namespace(let namespace):
            return namespace.attributes[item.pythonStr] != nil
        case .undefined:
            return false
        default:
            throw JinjaError.runtime("argument of type '\(container.typeName)' is not iterable")
        }
    }

    // MARK: - メソッド（"abc".strip() / dict.items() など）

    /// 呼び出さずに参照されたメソッド（関数値として返す）
    static func boundMethod(_ base: JinjaValue, _ name: String) -> JinjaValue? {
        guard hasMethod(base, name) else { return nil }
        return .function(JinjaFunction(name: name) { args, kwargs in
            try callMethod(base, name, args, kwargs) ?? .undefined
        })
    }

    private static func hasMethod(_ base: JinjaValue, _ name: String) -> Bool {
        switch base {
        case .string:
            return ["strip", "lstrip", "rstrip", "split", "rsplit", "splitlines", "startswith", "endswith", "upper",
                    "lower", "title", "capitalize", "replace", "find", "rfind", "count", "join", "isdigit", "isalpha",
                    "isalnum", "isspace", "isupper", "islower", "format", "removeprefix", "removesuffix"].contains(name)
        case .object:
            return ["items", "keys", "values", "get"].contains(name)
        case .array, .tuple:
            return ["count", "index"].contains(name)
        default:
            return false
        }
    }

    static func callMethod(_ base: JinjaValue, _ name: String, _ args: [JinjaValue],
                           _ kwargs: [(String, JinjaValue)]) throws -> JinjaValue? {
        switch base {
        case .string(let text):
            return try stringMethod(text, name, args, kwargs)
        case .object(let object):
            switch name {
            case "items": return .array(object.pairs.map { .tuple([.string($0.0), $0.1]) })
            case "keys": return .array(object.keys.map { .string($0) })
            case "values": return .array(object.pairs.map { $0.1 })
            case "get":
                guard let key = args.first else { return JinjaValue.none }
                return object[key.pythonStr] ?? (args.count > 1 ? args[1] : .none)
            default: return nil
            }
        case .array(let items), .tuple(let items):
            switch name {
            case "count": return .int(items.filter { $0.isEqual(to: args.first ?? .none) }.count)
            case "index":
                guard let index = items.firstIndex(where: { $0.isEqual(to: args.first ?? .none) }) else {
                    throw JinjaError.runtime("value is not in list")
                }
                return .int(index)
            case "append", "extend", "pop", "insert", "remove", "clear", "sort", "reverse":
                throw JinjaError.runtime("access to attribute '\(name)' of 'list' object is unsafe")
            default: return nil
            }
        default:
            return nil
        }
    }

    private static func isPythonSpace(_ scalar: Unicode.Scalar) -> Bool {
        scalar.properties.isWhitespace || (0x1C...0x1F).contains(scalar.value)
    }

    private static func strip(_ text: String, chars: String?, left: Bool, right: Bool) -> String {
        let set = chars.map { Set($0.unicodeScalars) }
        func shouldStrip(_ scalar: Unicode.Scalar) -> Bool {
            if let set { return set.contains(scalar) }
            return isPythonSpace(scalar)
        }
        var scalars = Array(text.unicodeScalars)[...]
        if left { while let first = scalars.first, shouldStrip(first) { scalars = scalars.dropFirst() } }
        if right { while let last = scalars.last, shouldStrip(last) { scalars = scalars.dropLast() } }
        var view = String.UnicodeScalarView()
        view.append(contentsOf: scalars)
        return String(view)
    }

    static func pythonSplit(_ text: String, separator: String?, maxSplit: Int) -> [String] {
        let scalars = Array(text.unicodeScalars)
        func make(_ slice: ArraySlice<Unicode.Scalar>) -> String {
            var view = String.UnicodeScalarView()
            view.append(contentsOf: slice)
            return String(view)
        }
        guard let separator, !separator.isEmpty else {
            // 空白の並びで分ける（空の要素は作らない）
            var parts: [String] = []
            var index = 0
            while index < scalars.count {
                while index < scalars.count && isPythonSpace(scalars[index]) { index += 1 }
                if index >= scalars.count { break }
                if maxSplit >= 0 && parts.count == maxSplit {
                    var end = scalars.count
                    while end > index && isPythonSpace(scalars[end - 1]) { end -= 1 }
                    parts.append(make(scalars[index..<end]))
                    return parts
                }
                let start = index
                while index < scalars.count && !isPythonSpace(scalars[index]) { index += 1 }
                parts.append(make(scalars[start..<index]))
            }
            return parts
        }
        let target = Array(separator.unicodeScalars)
        var parts: [String] = []
        var start = 0
        var index = 0
        while index + target.count <= scalars.count {
            if (maxSplit < 0 || parts.count < maxSplit) && Array(scalars[index..<(index + target.count)]) == target {
                parts.append(make(scalars[start..<index]))
                index += target.count
                start = index
            } else {
                index += 1
            }
        }
        parts.append(make(scalars[start...]))
        return parts
    }

    static func title(_ text: String) -> String {
        var result = ""
        var previousCased = false
        for character in text {
            let string = String(character)
            if character.isLetter {
                result += previousCased ? string.lowercased() : string.uppercased()
                previousCased = true
            } else {
                result += string
                previousCased = false
            }
        }
        return result
    }

    private static func stringMethod(_ text: String, _ name: String, _ args: [JinjaValue],
                                     _ kwargs: [(String, JinjaValue)]) throws -> JinjaValue? {
        func argument(_ index: Int, _ key: String) -> JinjaValue? {
            if index < args.count { return args[index] }
            return kwargs.first { $0.0 == key }?.1
        }
        func optionalString(_ value: JinjaValue?) -> String? {
            guard let value, !value.isNone, !value.isUndefined else { return nil }
            return value.renderedString
        }
        switch name {
        case "strip": return .string(strip(text, chars: optionalString(argument(0, "chars")), left: true, right: true))
        case "lstrip": return .string(strip(text, chars: optionalString(argument(0, "chars")), left: true, right: false))
        case "rstrip": return .string(strip(text, chars: optionalString(argument(0, "chars")), left: false, right: true))
        case "split":
            let maxSplit = Int(argument(1, "maxsplit")?.numericValue ?? -1)
            return .array(pythonSplit(text, separator: optionalString(argument(0, "sep")), maxSplit: maxSplit).map { .string($0) })
        case "rsplit":
            // 最大分割数なしなら split と同じ
            return .array(pythonSplit(text, separator: optionalString(argument(0, "sep")), maxSplit: -1).map { .string($0) })
        case "splitlines":
            return .array(text.split(omittingEmptySubsequences: false, whereSeparator: \.isNewline).map { .string(String($0)) }
                .dropLast(text.last?.isNewline == true ? 1 : 0).map { $0 })
        case "startswith", "endswith":
            var candidates: [String] = []
            if let value = argument(0, "prefix") {
                if case .tuple(let items) = value { candidates = items.map(\.renderedString) } else { candidates = [value.renderedString] }
            }
            let scalars = Array(text.unicodeScalars)
            let result = candidates.contains { candidate in
                let target = Array(candidate.unicodeScalars)
                return name == "startswith" ? scalars.starts(with: target) : scalars.reversed().starts(with: target.reversed())
            }
            return .bool(result)
        case "upper": return .string(text.uppercased())
        case "lower": return .string(text.lowercased())
        case "title": return .string(title(text))
        case "capitalize":
            guard let first = text.first else { return .string("") }
            return .string(String(first).uppercased() + text.dropFirst().lowercased())
        case "replace":
            guard let old = argument(0, "old")?.renderedString, let new = argument(1, "new")?.renderedString else { return .string(text) }
            let count = Int(argument(2, "count")?.numericValue ?? -1)
            if count < 0 { return .string((text as NSString).replacingOccurrences(of: old, with: new, options: .literal, range: NSRange(location: 0, length: (text as NSString).length))) }
            var result = text
            for _ in 0..<count {
                let range = (result as NSString).range(of: old, options: .literal)
                if range.location == NSNotFound { break }
                result = (result as NSString).replacingCharacters(in: range, with: new)
            }
            return .string(result)
        case "find", "rfind":
            guard let needle = argument(0, "sub")?.renderedString else { return .int(-1) }
            let options: NSString.CompareOptions = name == "find" ? [.literal] : [.literal, .backwards]
            let range = (text as NSString).range(of: needle, options: options)
            if range.location == NSNotFound { return .int(-1) }
            let prefix = (text as NSString).substring(to: range.location)
            return .int(prefix.unicodeScalars.count)
        case "count":
            guard let needle = argument(0, "sub")?.renderedString, !needle.isEmpty else { return .int(text.unicodeScalars.count + 1) }
            return .int(pythonSplit(text, separator: needle, maxSplit: -1).count - 1)
        case "join":
            let items = args.first?.iterationItems ?? []
            return .string(items.map(\.renderedString).joined(separator: text))
        case "isdigit": return .bool(!text.isEmpty && text.unicodeScalars.allSatisfy { $0.properties.numericType != nil })
        case "isalpha": return .bool(!text.isEmpty && text.unicodeScalars.allSatisfy { $0.properties.isAlphabetic })
        case "isalnum": return .bool(!text.isEmpty && text.unicodeScalars.allSatisfy { $0.properties.isAlphabetic || $0.properties.numericType != nil })
        case "isspace": return .bool(!text.isEmpty && text.unicodeScalars.allSatisfy(isPythonSpace))
        case "isupper": return .bool(text.uppercased() == text && text.lowercased() != text)
        case "islower": return .bool(text.lowercased() == text && text.uppercased() != text)
        case "removeprefix":
            let prefix = argument(0, "prefix")?.renderedString ?? ""
            let scalars = Array(text.unicodeScalars)
            let target = Array(prefix.unicodeScalars)
            return .string(scalars.starts(with: target) ? String(String.UnicodeScalarView(scalars.dropFirst(target.count))) : text)
        case "removesuffix":
            let suffix = argument(0, "suffix")?.renderedString ?? ""
            let scalars = Array(text.unicodeScalars)
            let target = Array(suffix.unicodeScalars)
            return .string(!target.isEmpty && scalars.reversed().starts(with: target.reversed())
                           ? String(String.UnicodeScalarView(scalars.dropLast(target.count))) : text)
        case "format":
            // "{}" と "{0}" だけ
            var result = ""
            var autoIndex = 0
            var iterator = Array(text.unicodeScalars)[...]
            while let scalar = iterator.first {
                iterator = iterator.dropFirst()
                if scalar == "{" {
                    if iterator.first == "{" { iterator = iterator.dropFirst(); result += "{"; continue }
                    var field = ""
                    while let next = iterator.first, next != "}" { field.unicodeScalars.append(next); iterator = iterator.dropFirst() }
                    iterator = iterator.dropFirst()
                    if let index = Int(field), index < args.count {
                        result += args[index].pythonStr
                    } else if field.isEmpty, autoIndex < args.count {
                        result += args[autoIndex].pythonStr
                        autoIndex += 1
                    } else if let value = kwargs.first(where: { $0.0 == field })?.1 {
                        result += value.pythonStr
                    }
                } else if scalar == "}" && iterator.first == "}" {
                    iterator = iterator.dropFirst()
                    result += "}"
                } else {
                    result.unicodeScalars.append(scalar)
                }
            }
            return .string(result)
        default:
            return nil
        }
    }

    // MARK: - テスト（is ...）

    static func applyTest(_ name: String, _ value: JinjaValue, _ args: [JinjaValue], interpreter: JinjaInterpreter) throws -> Bool {
        switch name {
        case "defined": return !value.isUndefined
        case "undefined": return value.isUndefined
        case "none": return value.isNone
        case "string":
            if case .string = value { return true }
            return false
        case "number":
            switch value { case .int, .double, .bool: return true; default: return false }
        case "integer":
            if case .int = value { return true }
            return false
        case "float":
            if case .double = value { return true }
            return false
        case "boolean":
            if case .bool = value { return true }
            return false
        case "true":
            if case .bool(true) = value { return true }
            return false
        case "false":
            if case .bool(false) = value { return true }
            return false
        case "mapping":
            if case .object = value { return true }
            return false
        case "sequence", "iterable":
            switch value { case .string, .array, .tuple, .object: return true; default: return false }
        case "callable":
            if case .function = value { return true }
            return false
        case "even": return Int(value.numericValue ?? 1) % 2 == 0
        case "odd": return Int(value.numericValue ?? 0) % 2 != 0
        case "divisibleby":
            guard let divisor = args.first?.numericValue, divisor != 0 else { return false }
            return Int(value.numericValue ?? 0) % Int(divisor) == 0
        case "eq", "equalto", "==": return value.isEqual(to: args.first ?? .undefined)
        case "ne", "!=": return !value.isEqual(to: args.first ?? .undefined)
        case "lt", "lessthan", "<": return try value.compare(args.first ?? .undefined) == .orderedAscending
        case "le", "<=": return try value.compare(args.first ?? .undefined) != .orderedDescending
        case "gt", "greaterthan", ">": return try value.compare(args.first ?? .undefined) == .orderedDescending
        case "ge", ">=": return try value.compare(args.first ?? .undefined) != .orderedAscending
        case "in": return try contains(args.first ?? .undefined, value)
        case "sameas":
            if case .namespace(let a) = value, case .namespace(let b)? = args.first { return a === b }
            return value.isEqual(to: args.first ?? .undefined)
        case "upper":
            if case .string(let text) = value { return text.uppercased() == text }
            return false
        case "lower":
            if case .string(let text) = value { return text.lowercased() == text }
            return false
        case "escaped":
            return false
        default:
            throw JinjaError.runtime("no test named '\(name)'")
        }
    }

    // MARK: - フィルタ（| ...）

    static func applyFilter(_ name: String, _ value: JinjaValue, _ args: [JinjaValue], _ kwargs: [(String, JinjaValue)],
                            interpreter: JinjaInterpreter) throws -> JinjaValue {
        func argument(_ index: Int, _ key: String) -> JinjaValue? {
            if let keyword = kwargs.first(where: { $0.0 == key }) { return keyword.1 }
            return index >= 0 && index < args.count ? args[index] : nil
        }
        func items(_ value: JinjaValue) throws -> [JinjaValue] {
            guard let items = value.iterationItems else {
                throw JinjaError.runtime("'\(value.typeName)' object is not iterable")
            }
            return items
        }
        func attributePath(_ item: JinjaValue, _ path: String) throws -> JinjaValue {
            var current = item
            for part in path.split(separator: ".") {
                if let number = Int(part) {
                    current = try subscriptValue(current, .int(number))
                } else {
                    current = try subscriptValue(current, .string(String(part)))
                }
            }
            return current
        }
        switch name {
        case "length", "count":
            switch value {
            case .string(let text): return .int(text.unicodeScalars.count)
            case .array(let list), .tuple(let list): return .int(list.count)
            case .object(let object): return .int(object.count)
            case .undefined: return .int(0)
            default: throw JinjaError.runtime("object of type '\(value.typeName)' has no len()")
            }
        case "tojson":
            let indent = argument(1, "indent").flatMap { $0.numericValue }.map { Int($0) }
            let ensureASCII = argument(0, "ensure_ascii")?.truthy ?? false
            return .string(try value.toJSON(indent: indent, ensureASCII: ensureASCII))
        case "trim":
            let chars = argument(0, "chars").flatMap { $0.isNone || $0.isUndefined ? nil : $0.renderedString }
            return .string(strip(value.renderedString, chars: chars, left: true, right: true))
        case "safe", "string":
            if name == "string" { return .string(value.isUndefined ? "" : value.pythonStr) }
            return value
        case "e", "escape", "forceescape":
            let escaped = value.renderedString
                .replacingOccurrences(of: "&", with: "&amp;").replacingOccurrences(of: "<", with: "&lt;")
                .replacingOccurrences(of: ">", with: "&gt;").replacingOccurrences(of: "\"", with: "&#34;")
                .replacingOccurrences(of: "'", with: "&#39;")
            return .string(escaped)
        case "upper": return .string(value.renderedString.uppercased())
        case "lower": return .string(value.renderedString.lowercased())
        case "capitalize":
            let text = value.renderedString
            guard let first = text.first else { return .string("") }
            return .string(String(first).uppercased() + text.dropFirst().lowercased())
        case "title": return .string(title(value.renderedString))
        case "first": return try items(value).first ?? .undefined
        case "last": return try items(value).last ?? .undefined
        case "list": return .array(try items(value))
        case "items":
            guard case .object(let object) = value else { return .array([]) }
            return .array(object.pairs.map { .tuple([.string($0.0), $0.1]) })
        case "dictsort":
            guard case .object(let object) = value else { return .array([]) }
            return .array(object.pairs.sorted { $0.0 < $1.0 }.map { .tuple([.string($0.0), $0.1]) })
        case "join":
            let separator = argument(0, "d")?.renderedString ?? ""
            var list = try items(value)
            if let attribute = argument(1, "attribute")?.renderedString {
                list = try list.map { try attributePath($0, attribute) }
            }
            return .string(list.map(\.renderedString).joined(separator: separator))
        case "default", "d":
            let fallback = argument(0, "default_value") ?? .string("")
            let boolean = argument(1, "boolean")?.truthy ?? false
            if value.isUndefined || (boolean && !value.truthy) { return fallback }
            return value
        case "int":
            switch value {
            case .int: return value
            case .double(let number): return .int(Int(number))
            case .bool(let flag): return .int(flag ? 1 : 0)
            case .string(let text):
                let trimmed = text.trimmingCharacters(in: .whitespaces)
                if let number = Int(trimmed) { return .int(number) }
                if let number = Double(trimmed) { return .int(Int(number)) }
                return argument(0, "default") ?? .int(0)
            default: return argument(0, "default") ?? .int(0)
            }
        case "float":
            if let number = value.numericValue { return .double(number) }
            if case .string(let text) = value, let number = Double(text.trimmingCharacters(in: .whitespaces)) { return .double(number) }
            return argument(0, "default") ?? .double(0)
        case "abs":
            switch value {
            case .int(let number): return .int(abs(number))
            case .double(let number): return .double(abs(number))
            default: throw JinjaError.runtime("bad operand type for abs()")
            }
        case "round":
            let precision = Int(argument(0, "precision")?.numericValue ?? 0)
            let method = argument(1, "method")?.renderedString ?? "common"
            let number = value.numericValue ?? 0
            let scale = pow(10, Double(precision))
            let rule: FloatingPointRoundingRule = method == "ceil" ? .up : method == "floor" ? .down : .toNearestOrEven
            return .double((number * scale).rounded(rule) / scale)
        case "reverse":
            if case .string(let text) = value { return .string(String(text.reversed())) }
            return .array(try items(value).reversed())
        case "sort":
            let reverse = argument(0, "reverse")?.truthy ?? false
            let attribute = argument(2, "attribute")?.renderedString
            var list = try items(value)
            var failure: Error?
            list.sort { a, b in
                do {
                    let left = try attribute.map { try attributePath(a, $0) } ?? a
                    let right = try attribute.map { try attributePath(b, $0) } ?? b
                    let result = try left.compare(right)
                    return reverse ? result == .orderedDescending : result == .orderedAscending
                } catch {
                    failure = error
                    return false
                }
            }
            if let failure { throw failure }
            return .array(list)
        case "unique":
            var result: [JinjaValue] = []
            for item in try items(value) where !result.contains(where: { $0.isEqual(to: item) }) {
                result.append(item)
            }
            return .array(result)
        case "sum":
            var total: JinjaValue = argument(1, "start") ?? .int(0)
            for item in try items(value) { total = try binary("+", total, item) }
            return total
        case "min", "max":
            let list = try items(value)
            guard var best = list.first else { return .undefined }
            for item in list.dropFirst() {
                let result = try item.compare(best)
                if (name == "min" && result == .orderedAscending) || (name == "max" && result == .orderedDescending) { best = item }
            }
            return best
        case "replace":
            let old = argument(0, "old")?.renderedString ?? ""
            let new = argument(1, "new")?.renderedString ?? ""
            return try stringMethod(value.renderedString, "replace", [.string(old), .string(new)] + (argument(2, "count").map { [$0] } ?? []), []) ?? value
        case "indent":
            let width = Int(argument(0, "width")?.numericValue ?? 4)
            let first = argument(1, "first")?.truthy ?? false
            let blank = argument(2, "blank")?.truthy ?? false
            let pad = String(repeating: " ", count: width)
            let lines = value.renderedString.components(separatedBy: "\n")
            var result: [String] = []
            for (index, line) in lines.enumerated() {
                if index == 0 && !first { result.append(line); continue }
                result.append(line.isEmpty && !blank ? line : pad + line)
            }
            return .string(result.joined(separator: "\n"))
        case "wordcount":
            return .int(pythonSplit(value.renderedString, separator: nil, maxSplit: -1).count)
        case "format":
            return .string(try percentFormat(value.renderedString, .tuple(args)))
        case "attr":
            return try subscriptValue(value, args.first ?? .undefined)
        case "map":
            let list = try items(value)
            if let attribute = argument(-1, "attribute")?.renderedString {
                let fallback = kwargs.first { $0.0 == "default" }?.1
                return .array(try list.map { item in
                    let result = try attributePath(item, attribute)
                    if result.isUndefined, let fallback { return fallback }
                    return result
                })
            }
            guard let filterName = args.first?.renderedString else { return .array(list) }
            let rest = Array(args.dropFirst())
            return .array(try list.map { try applyFilter(filterName, $0, rest, [], interpreter: interpreter) })
        case "select", "reject":
            let list = try items(value)
            let keep = name == "select"
            guard let testName = args.first?.renderedString else {
                return .array(list.filter { $0.truthy == keep })
            }
            let rest = Array(args.dropFirst())
            return .array(try list.filter { try applyTest(testName, $0, rest, interpreter: interpreter) == keep })
        case "selectattr", "rejectattr":
            let list = try items(value)
            let keep = name == "selectattr"
            guard let attribute = args.first?.renderedString else { return .array(list) }
            if args.count < 2 {
                return .array(try list.filter { try attributePath($0, attribute).truthy == keep })
            }
            let testName = args[1].renderedString
            let rest = Array(args.dropFirst(2))
            return .array(try list.filter { try applyTest(testName, try attributePath($0, attribute), rest, interpreter: interpreter) == keep })
        case "batch":
            let size = max(1, Int(args.first?.numericValue ?? 1))
            let list = try items(value)
            return .array(stride(from: 0, to: list.count, by: size).map { .array(Array(list[$0..<min($0 + size, list.count)])) })
        case "center":
            let width = Int(args.first?.numericValue ?? 80)
            let text = value.renderedString
            let padding = max(0, width - text.count)
            let left = padding / 2
            return .string(String(repeating: " ", count: left) + text + String(repeating: " ", count: padding - left))
        case "truncate":
            let length = Int(argument(0, "length")?.numericValue ?? 255)
            let text = value.renderedString
            return .string(text.count <= length ? text : String(text.prefix(max(0, length - 3))) + "...")
        default:
            throw JinjaError.runtime("no filter named '\(name)'")
        }
    }
}
