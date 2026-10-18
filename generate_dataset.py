
import numpy as np
import pandas as pd


def generate_sim_swap_dataset(
    n_samples=10000,
    fraud_ratio=0.12,
    random_state=42,
):
    """
    Generate synthetic SIM-swapping fraud detection data.

    This is simulated data for prototyping, not real customer data.
    Fraud labels are generated from simulated behavioral patterns.
    """

    if n_samples < 1:
        raise ValueError("n_samples must be at least 1.")

    if not 0 <= fraud_ratio <= 1:
        raise ValueError("fraud_ratio must be between 0 and 1.")

    rng = np.random.default_rng(random_state)

    # Generate the target label.
    # 0 = legitimate transaction
    # 1 = simulated fraudulent transaction
    is_fraud = rng.binomial(1, fraud_ratio, n_samples)

    # Time elapsed between SIM change and transaction, in hours.
    # Simulated fraud is more concentrated shortly after SIM changes.
    hours_since_sim_change = np.where(
        is_fraud == 1,
        rng.exponential(scale=8, size=n_samples),
        rng.exponential(scale=240, size=n_samples),
    )
    hours_since_sim_change = np.clip(
        hours_since_sim_change, 0, 720
    )

    # Whether the transaction comes from a new device.
    new_device = np.where(
        is_fraud == 1,
        rng.binomial(1, 0.72, n_samples),
        rng.binomial(1, 0.12, n_samples),
    )

    # Failed login attempts during the recent observation window.
    failed_logins = np.where(
        is_fraud == 1,
        rng.poisson(3.0, n_samples),
        rng.poisson(0.5, n_samples),
    )
    failed_logins = np.clip(failed_logins, 0, 15)

    # Transaction amount in the chosen currency units.
    # Use INR for the dashboard demonstration.
    normal_amount = rng.lognormal(
        mean=np.log(1200), sigma=0.8, size=n_samples
    )
    suspicious_amount = rng.lognormal(
        mean=np.log(6500), sigma=0.9, size=n_samples
    )
    transaction_amount = np.where(
        is_fraud == 1,
        suspicious_amount,
        normal_amount,
    )
    transaction_amount = np.clip(
        transaction_amount, 10, 200000
    ).round(2)

    # Average historical transaction amount for the account.
    average_transaction_amount = rng.lognormal(
        mean=np.log(1500), sigma=0.7, size=n_samples
    )
    average_transaction_amount = np.clip(
        average_transaction_amount, 10, 100000
    ).round(2)

    # Number of transactions in the preceding 24 hours.
    transaction_count_24h = np.where(
        is_fraud == 1,
        rng.poisson(5.0, n_samples),
        rng.poisson(2.0, n_samples),
    )
    transaction_count_24h = np.clip(
        transaction_count_24h, 0, 30
    )

    # Whether the transaction originates from an unusual location.
    unusual_location = np.where(
        is_fraud == 1,
        rng.binomial(1, 0.55, n_samples),
        rng.binomial(1, 0.08, n_samples),
    )

    # Whether account credentials were recently changed.
    recent_credential_change = np.where(
        is_fraud == 1,
        rng.binomial(1, 0.45, n_samples),
        rng.binomial(1, 0.05, n_samples),
    )

    # Whether SIM-swap status is known to have been detected.
    # This is a contextual signal, not proof of fraud.
    sim_change_detected = np.where(
        is_fraud == 1,
        rng.binomial(1, 0.85, n_samples),
        rng.binomial(1, 0.10, n_samples),
    )

    # A synthetic, non-identifying account reference.
    account_id = [
        f"ACC{i:06d}" for i in range(1, n_samples + 1)
    ]

    df = pd.DataFrame({
        "account_id": account_id,
        "hours_since_sim_change": hours_since_sim_change.round(2),
        "sim_change_detected": sim_change_detected,
        "new_device": new_device,
        "failed_logins": failed_logins,
        "transaction_amount": transaction_amount,
        "average_transaction_amount": average_transaction_amount,
        "transaction_count_24h": transaction_count_24h,
        "unusual_location": unusual_location,
        "recent_credential_change": recent_credential_change,
        "is_fraud": is_fraud,
    })

    # Shuffle rows so fraud and legitimate transactions are mixed.
    df = df.sample(
        frac=1, random_state=random_state
    ).reset_index(drop=True)

    return df


if __name__ == "__main__":
    dataset = generate_sim_swap_dataset(
        n_samples=10000,
        fraud_ratio=0.12,
        random_state=42,
    )

    # Save to the data folder.
    from pathlib import Path

    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "sim_swap_dataset.csv"
    dataset.to_csv(output_file, index=False)

    print("Synthetic dataset generated successfully!")
    print(f"Saved to: {output_file.resolve()}")
    print(f"Dataset shape: {dataset.shape}")
    print("\nFirst five rows:")
    print(dataset.head().to_string(index=False))

    print("\nFraud label counts:")
    print(dataset["is_fraud"].value_counts())

    print("\nFraud label proportions:")
    print(dataset["is_fraud"].value_counts(normalize=True).round(3))
