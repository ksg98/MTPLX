import XCTest
@testable import MTPLXAppCore

/// The app must decode a Splash-backed daemon with the same DTOs it uses for
/// the MLX one. A missing non-optional field fails the whole decode and blanks
/// the dashboard, so these fixtures are real responses captured from a live
/// `mtplx serve --engine splash`, not hand-written approximations.
final class SplashBridgeDecodingTests: XCTestCase {

    private func fixture(_ name: String) throws -> Data {
        let url = try XCTUnwrap(
            Bundle.module.url(forResource: name, withExtension: "json"),
            "missing fixture \(name).json"
        )
        return try Data(contentsOf: url)
    }

    private var decoder: JSONDecoder { MTPLXAPIClient.makeDefaultDecoder() }

    func testHealthDecodes() throws {
        let health = try decoder.decode(HealthPayload.self, from: fixture("splash_health"))
        XCTAssertTrue(health.ok)
        XCTAssertEqual(health.model, "incoai/Qwen3.8-27B-Splash")
        XCTAssertTrue(health.modelPath.hasSuffix("incoai/Qwen3.8-27B-Splash"))
        XCTAssertEqual(health.contextWindow, 131072)
        // What the supervisor verifies a launch against.
        XCTAssertEqual(health.startup?.launchId, "verify-launch-1")
        // Splash drafts with DFlash 2, which is not MTPLX's MTP.
        XCTAssertFalse(health.mtpEnabled)
        XCTAssertEqual(health.startup?.modelControls?.backendID, "splash")
    }

    func testSnapshotDecodesSoTheDashboardRenders() throws {
        let snapshot = try decoder.decode(
            DashboardSnapshot.self, from: fixture("splash_mtplx_snapshot")
        )
        XCTAssertEqual(snapshot.modelId, "incoai/Qwen3.8-27B-Splash")
        XCTAssertEqual(snapshot.contextWindow, 131072)
        XCTAssertTrue(snapshot.mem.ok)
        XCTAssertGreaterThan(snapshot.uptimeS, 0)
    }

    /// Captured after two real streamed generations, so the rows the Live tab
    /// draws from are populated rather than structurally valid and empty.
    func testSnapshotCarriesTheSpeedsAndTheMinMaxRow() throws {
        let snapshot = try decoder.decode(
            DashboardSnapshot.self, from: fixture("splash_mtplx_snapshot")
        )
        let rolling = snapshot.rolling
        XCTAssertGreaterThanOrEqual(rolling.count, 2)
        let low = try XCTUnwrap(rolling.min), high = try XCTUnwrap(rolling.max)
        XCTAssertGreaterThan(low, 0)
        XCTAssertGreaterThanOrEqual(high, low)
        XCTAssertNotNil(rolling.mean)
        XCTAssertNotNil(rolling.p95)
        XCTAssertGreaterThan(rolling.stickyAllTimeMax, 0)
        XCTAssertFalse(rolling.history.isEmpty)

        // The newest request is last; its envelope is what `latest` becomes.
        let newest = try XCTUnwrap(snapshot.recent.last)
        XCTAssertGreaterThan(try XCTUnwrap(newest.decodeTokS), 0)
        XCTAssertGreaterThan(try XCTUnwrap(newest.prefillTokS), 0)
        XCTAssertNotNil(newest.ttftS)

        // The memory tile shows the engine, not just its KV pages.
        XCTAssertGreaterThan(try XCTUnwrap(snapshot.mem.activeMemoryBytes), 10_000_000_000)
        XCTAssertGreaterThan(snapshot.lifetime.completionTokensTotal, 0)
    }

    /// Avg Prefill, Cached and Context read optional snapshot blocks, which
    /// decode as nil — and render as a dash — when the server omits them.
    func testOptionalBlocksBehindThePrefillCachedAndContextTiles() throws {
        let snapshot = try decoder.decode(
            DashboardSnapshot.self, from: fixture("splash_mtplx_snapshot")
        )
        let rates = try XCTUnwrap(snapshot.prefillRates, "Avg Prefill reads prefill_rates")
        XCTAssertGreaterThan(rates.tokens, 0)
        XCTAssertGreaterThan(try XCTUnwrap(rates.averageTokS), 0)
        XCTAssertNotNil(rates.peakTokS)

        // Second turn of a conversation: most of the prompt was reused.
        let newest = try XCTUnwrap(snapshot.recent.last)
        XCTAssertGreaterThan(newest.values["cached_tokens"]?.intValue ?? 0, 0)
        XCTAssertEqual(newest.values["session_cache_hit"]?.boolValue, true)
        XCTAssertGreaterThan(snapshot.lifetime.cachedTokensTotal, 0)

        // Both turns belong to one conversation, so one session row.
        XCTAssertEqual(snapshot.sessions.count, 1)
        let row = try XCTUnwrap(snapshot.sessions.sessions.first)
        XCTAssertGreaterThan(row.prefixLen, 0)
        XCTAssertGreaterThan(row.bytes, 0)
        XCTAssertNotNil(snapshot.latest)
    }

    func testCapabilitiesDecodeAndDeclareWhatSplashLacks() throws {
        let capabilities = try decoder.decode(
            AppCapabilities.self, from: fixture("splash_mtplx_app_capabilities")
        )
        XCTAssertTrue(capabilities.ok)
        XCTAssertEqual(capabilities.features["mtp"], false)
        XCTAssertEqual(capabilities.features["kv_quantization"], false)
        XCTAssertEqual(capabilities.features["chat"], true)
        XCTAssertGreaterThan(capabilities.snapshotInterval.defaultMs, 0)
    }

    func testSettingsFreezeKVQuantWithAReason() throws {
        let settings = try decoder.decode(
            MutableSettings.self, from: fixture("splash_mtplx_settings")
        )
        let policy = try XCTUnwrap(settings.kvQuantPolicy)
        XCTAssertFalse(policy.supported)
        XCTAssertEqual(policy.modes, ["q8"])
        XCTAssertNotNil(policy.disabledReason, "a locked control must say why")
    }

    func testRemainingContractEndpointsDecode() throws {
        let sessions = try decoder.decode(
            SessionsPayload.self, from: fixture("splash_admin_sessions")
        )
        XCTAssertEqual(sessions.count, sessions.sessions.count)
        let prefill = try decoder.decode(
            PrefillHistoryPayload.self, from: fixture("splash_mtplx_prefill_history")
        )
        XCTAssertFalse(prefill.history.isEmpty, "one row per completed request")
        let models = try decoder.decode(ModelsResponse.self, from: fixture("splash_models"))
        XCTAssertEqual(models.data.first?.id, "incoai/Qwen3.8-27B-Splash")
    }
}
