import SwiftData
import SwiftUI

@main
struct LocalLLMApp: App {
    @State private var settings = AppSettings()
    @State private var library = ModelLibrary()
    @State private var chat = ChatController()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environment(settings)
                .environment(library)
                .environment(chat)
        }
        .modelContainer(for: [Conversation.self, StoredMessage.self])
    }
}
