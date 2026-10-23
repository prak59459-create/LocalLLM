import SwiftData
import SwiftUI

@main
struct LocalLLMApp: App {
    @State private var settings = AppSettings()
    @State private var library = ModelLibrary()
    @State private var chat = ChatController()
    @State private var knowledge = KnowledgeStore()
    @State private var studio = MiniLLMStudio()
    @State private var connectors = ConnectorStore()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environment(settings)
                .environment(library)
                .environment(chat)
                .environment(knowledge)
                .environment(studio)
                .environment(connectors)
        }
        .modelContainer(for: [Conversation.self, StoredMessage.self, Assistant.self])
    }
}
