import numpy as np

from italian_real_estate.ml.model_evaluator import evaluate_model


def test_mape_is_computed_on_original_scale_for_log_target():
    actual_price = np.array([100.0, 200.0])
    predicted_price = np.array([110.0, 180.0])

    metrics = evaluate_model(
        np.log1p(actual_price),
        np.log1p(predicted_price),
        log_transformed_target=True,
    )

    np.testing.assert_allclose(metrics["mape"], 0.1, rtol=1e-12)


def test_mape_can_be_computed_for_untransformed_target():
    metrics = evaluate_model(
        np.array([100.0, 200.0]),
        np.array([110.0, 180.0]),
        log_transformed_target=False,
    )

    np.testing.assert_allclose(metrics["mape"], 0.1, rtol=1e-12)
