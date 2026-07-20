import XCTest
@testable import oMLX

final class MenubarControllerModelsTests: XCTestCase {

    // MARK: - toggleState

    func testToggleStateUnloadingTakesPrecedenceOverLoaded() {
        let model = makeModel("a", loaded: true)
        XCTAssertEqual(
            MenubarController.toggleState(for: model, unloading: ["a"]),
            .unloading
        )
    }

    func testToggleStateUnloadingTakesPrecedenceOverLoading() {
        let model = makeModel("a", isLoading: true)
        XCTAssertEqual(
            MenubarController.toggleState(for: model, unloading: ["a"]),
            .unloading
        )
    }

    func testToggleStateLoadingWhenModelIsLoading() {
        let model = makeModel("a", isLoading: true)
        XCTAssertEqual(
            MenubarController.toggleState(for: model, unloading: []),
            .loading
        )
    }

    func testToggleStateUnloadWhenLoaded() {
        let model = makeModel("a", loaded: true)
        XCTAssertEqual(
            MenubarController.toggleState(for: model, unloading: []),
            .unload
        )
    }

    func testToggleStateLoadWhenIdle() {
        let model = makeModel("a")
        XCTAssertEqual(
            MenubarController.toggleState(for: model, unloading: []),
            .load
        )
    }

    // MARK: - reconcileUnloading

    func testReconcileUnloadingKeepsOnlyStillLoadedIDs() {
        let models = [makeModel("a", loaded: true), makeModel("b", loaded: false)]
        XCTAssertEqual(
            MenubarController.reconcileUnloading(["a", "b"], against: models),
            ["a"]
        )
    }

    func testReconcileUnloadingDropsUnknownIDs() {
        let models = [makeModel("a", loaded: true)]
        XCTAssertEqual(
            MenubarController.reconcileUnloading(["x"], against: models),
            []
        )
    }

    func testReconcileUnloadingEmptyStaysEmpty() {
        let models = [makeModel("a", loaded: true)]
        XCTAssertEqual(
            MenubarController.reconcileUnloading([], against: models),
            []
        )
    }

    // MARK: - modelMenuTitle

    func testModelMenuTitlePrefixesLoadedModel() {
        XCTAssertEqual(
            MenubarController.modelMenuTitle(for: makeModel("org/model", loaded: true)),
            "✅ org/model"
        )
    }

    func testModelMenuTitlePlainForUnloadedModel() {
        XCTAssertEqual(
            MenubarController.modelMenuTitle(for: makeModel("org/model", loaded: false)),
            "org/model"
        )
    }

    // MARK: - ModelDTO.sizeLabel

    func testSizeLabelWhileLoadingUsesEstimated() {
        let model = makeModel("a", isLoading: true,
                              estimatedSizeFormatted: "10 GB", actualSizeFormatted: "9 GB")
        XCTAssertEqual(model.sizeLabel, "10 GB")
    }

    func testSizeLabelLoadedWithDifferingActualShowsBoth() {
        let model = makeModel("a", loaded: true,
                              estimatedSizeFormatted: "10 GB", actualSizeFormatted: "9 GB")
        XCTAssertEqual(model.sizeLabel, "~9 GB obs / 10 GB est")
    }

    func testSizeLabelLoadedWithEqualActualShowsObservedOnly() {
        let model = makeModel("a", loaded: true,
                              estimatedSizeFormatted: "9 GB", actualSizeFormatted: "9 GB")
        XCTAssertEqual(model.sizeLabel, "~9 GB obs")
    }

    func testSizeLabelLoadedWithoutActualUsesEstimated() {
        let model = makeModel("a", loaded: true, estimatedSizeFormatted: "10 GB")
        XCTAssertEqual(model.sizeLabel, "10 GB")
    }

    func testSizeLabelEmptyWhenNoSizes() {
        XCTAssertEqual(makeModel("a", loaded: true).sizeLabel, "")
    }

    // MARK: - Helpers

    private func makeModel(
        _ id: String,
        loaded: Bool = false,
        isLoading: Bool = false,
        estimatedSizeFormatted: String? = nil,
        actualSizeFormatted: String? = nil
    ) -> ModelDTO {
        ModelDTO(
            id: id,
            displayName: nil,
            modelPath: nil,
            loaded: loaded,
            isLoading: isLoading,
            estimatedSize: 0,
            estimatedSizeFormatted: estimatedSizeFormatted,
            actualSize: nil,
            actualSizeFormatted: actualSizeFormatted,
            pinned: nil,
            isDefault: nil,
            isFavorite: nil,
            engineType: nil,
            modelType: nil,
            configModelType: nil,
            thinkingDefault: nil,
            dflashCompatible: nil,
            dflashCompatibilityReason: nil,
            dflashSsdCacheAvailable: nil,
            mtpCompatible: nil,
            mtpCompatibilityReason: nil,
            settings: nil
        )
    }
}
