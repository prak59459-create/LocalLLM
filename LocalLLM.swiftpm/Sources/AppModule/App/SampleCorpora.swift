import Foundation

/// ミニLLM工房で使うサンプルの学習データ。
enum SampleCorpora {
    /// 夏目漱石「吾輩は猫である」冒頭（著作権の保護期間が終了した作品）
    static let wagahai = """
    吾輩は猫である。名前はまだ無い。
    どこで生れたかとんと見当がつかぬ。何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。吾輩はここで始めて人間というものを見た。しかもあとで聞くとそれは書生という人間中で一番獰悪な種族であったそうだ。この書生というのは時々我々を捕えて煮て食うという話である。しかしその当時は何という考もなかったから別段恐しいとも思わなかった。ただ彼の掌に載せられてスーと持ち上げられた時何だかフワフワした感じがあったばかりである。掌の上で少し落ちついて書生の顔を見たのがいわゆる人間というものの見始であろう。この時妙なものだと思った感じが今でも残っている。第一毛をもって装飾されべきはずの顔がつるつるしてまるで薬缶だ。その後猫にもだいぶ逢ったがこんな片輪には一度も出会わした事がない。のみならず顔の真中があまりに突起している。そうしてその穴の中から時々ぷうぷうと煙を吹く。どうも咽せぽくて実に弱った。これが人間の飲む煙草というものである事はようやくこの頃知った。
    この書生の掌の裏でしばらくはよい心持に坐っておったが、しばらくすると非常な速力で運転し始めた。書生が動くのか自分だけが動くのか分らないが無暗に眼が廻る。胸が悪くなる。到底助からないと思っていると、どさりと音がして眼から火が出た。それまでは記憶しているがあとは何の事やらいくら考え出そうとしても分らない。
    ふと気が付いて見ると書生はいない。たくさんおった兄弟が一疋も見えぬ。肝心の母親さえ姿を隠してしまった。その上今までの所とは違って無暗に明るい。眼を明いていられぬくらいだ。はてな何でも容子がおかしいと、のそのそ這い出して見ると非常に痛い。吾輩は藁の上から急に笹原の中へ棄てられたのである。
    ようやくの思いで笹原を這い出すと向うに大きな池がある。吾輩は池の前に坐ってどうしたらよかろうと考えて見た。別にこれという分別も出ない。しばらくして泣いたら書生がまた迎に来てくれるかと考え付いた。ニャー、ニャーと試みにやって見たが誰も来ない。そのうち池の上をさらさらと風が渡って日が暮れかかる。腹が非常に減って来た。泣きたくても声が出ない。仕方がない、何でもよいから食物のある所まであるこうと決心をしてそろりそろりと池を左りに廻り始めた。
    """

    /// Swift のコード例（「自分専用コーディング AI」を小さく体験する用）
    static let swiftCode = """
    import SwiftUI

    struct Item: Identifiable, Hashable {
        let id = UUID()
        var title: String
        var isDone: Bool = false
    }

    @Observable
    final class TodoStore {
        var items: [Item] = []

        func add(_ title: String) {
            let trimmed = title.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { return }
            items.append(Item(title: trimmed))
        }

        func toggle(_ item: Item) {
            guard let index = items.firstIndex(of: item) else { return }
            items[index].isDone.toggle()
        }

        func remove(at offsets: IndexSet) {
            items.remove(atOffsets: offsets)
        }
    }

    struct ContentView: View {
        @State private var store = TodoStore()
        @State private var draft = ""

        var body: some View {
            NavigationStack {
                List {
                    Section {
                        HStack {
                            TextField("New item", text: $draft)
                            Button("Add") {
                                store.add(draft)
                                draft = ""
                            }
                        }
                    }
                    Section {
                        ForEach(store.items) { item in
                            Button {
                                store.toggle(item)
                            } label: {
                                HStack {
                                    Image(systemName: item.isDone ? "checkmark.circle.fill" : "circle")
                                    Text(item.title)
                                }
                            }
                        }
                        .onDelete(perform: store.remove)
                    }
                }
                .navigationTitle("Todo")
            }
        }
    }

    func fibonacci(_ n: Int) -> [Int] {
        guard n > 0 else { return [] }
        var numbers = [0, 1]
        while numbers.count < n {
            numbers.append(numbers[numbers.count - 1] + numbers[numbers.count - 2])
        }
        return Array(numbers.prefix(n))
    }

    func isPrime(_ value: Int) -> Bool {
        guard value >= 2 else { return false }
        var divisor = 2
        while divisor * divisor <= value {
            if value % divisor == 0 { return false }
            divisor += 1
        }
        return true
    }

    extension Array where Element: Comparable {
        func insertionSorted() -> [Element] {
            var result = self
            guard result.count > 1 else { return result }
            for index in 1..<result.count {
                var current = index
                while current > 0 && result[current - 1] > result[current] {
                    result.swapAt(current - 1, current)
                    current -= 1
                }
            }
            return result
        }
    }
    """

    /// ダウンロードできる大きめのデータ
    static let tinyShakespeareURL = URL(string: "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")!
}
