import Foundation
import UIKit

/// アプリが前に出ているか。iOS は裏に回ったアプリに GPU を使わせないので、そのあいだに失敗した GPU の計算は、
/// CPU に切り替えずに前に戻るのを待ってからやり直す（どのスレッドからでも読める）。
enum AppActivity {
    private static let lock = NSLock()
    private static var background = false
    private static var observers: [NSObjectProtocol] = []

    /// 通知を受け取り始める（起動時に 1 回）
    @MainActor
    static func start() {
        lock.lock()
        let started = !observers.isEmpty
        lock.unlock()
        guard !started else { return }
        set(background: UIApplication.shared.applicationState == .background)
        let center = NotificationCenter.default
        let entered = center.addObserver(forName: UIApplication.didEnterBackgroundNotification, object: nil, queue: nil) { _ in
            set(background: true)
        }
        let returned = center.addObserver(forName: UIApplication.willEnterForegroundNotification, object: nil, queue: nil) { _ in
            set(background: false)
        }
        lock.lock()
        observers = [entered, returned]
        lock.unlock()
    }

    /// 前に出ている（GPU を使える）か
    static var isActive: Bool {
        lock.lock()
        defer { lock.unlock() }
        return !background
    }

    /// 前に戻るまで待つ（取り消されたら CancellationError）
    static func waitUntilActive() async throws {
        while !isActive {
            try await Task.sleep(nanoseconds: 500_000_000)
        }
    }

    private static func set(background value: Bool) {
        lock.lock()
        background = value
        lock.unlock()
    }
}
