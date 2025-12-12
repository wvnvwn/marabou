"""
Marabou-based Neural Network Verification

Implements formal verification using the Marabou SMT solver.
"""

import torch
import numpy as np
import time

# Optional Marabou import - allows training without Marabou installed
try:
    from maraboupy import Marabou
    MARABOU_AVAILABLE = True
except ImportError:
    MARABOU_AVAILABLE = False
    Marabou = None
    print("⚠ Warning: Marabou not installed. Verification will be skipped.")
    print("  Training and ONNX conversion will still work.")
    print("  Install Marabou later with: pip install maraboupy")


class MarabouVerifier:
    """
    Wrapper class for Marabou verification tool.

    Provides high-level interface for verifying neural network properties.
    """

    def __init__(self, onnx_path, timeout=60):
        """
        Args:
            onnx_path: Path to ONNX model file
            timeout: Timeout in seconds for each verification query
        """
        self.onnx_path = onnx_path
        self.timeout = timeout
        self.network = None
        self._load_network()

    def _load_network(self):
        """Load ONNX network into Marabou."""
        if not MARABOU_AVAILABLE:
            print("✗ Marabou not available. Skipping network loading.")
            return

        try:
            self.network = Marabou.read_onnx(self.onnx_path)
            print(f"✓ Model loaded in Marabou from {self.onnx_path}")
            print(f"  Input variables: {len(self.network.inputVars[0])}")
            print(f"  Output variables: {len(self.network.outputVars[0])}")
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            raise

    def verify_local_robustness(self, image, true_label, epsilon=0.02,
                               pixel_bounds=(-5.0, 5.0), verbose=True):
        """
        Verify local robustness property for a given input.

        Args:
            image: Input image tensor
            true_label: True class label
            epsilon: Perturbation bound
            pixel_bounds: Valid range for pixel values
            verbose: Whether to print progress

        Returns:
            Tuple of (is_robust, counterexample_class, verification_time)
        """
        if not MARABOU_AVAILABLE or self.network is None:
            if verbose:
                print(f"\n{'='*60}")
                print(f"⚠ Marabou not available - Skipping verification")
                print(f"{'='*60}")
            return None, None, 0.0

        if verbose:
            print(f"\n{'='*60}")
            print(f"Local Robustness Verification")
            print(f"{'='*60}")
            print(f"True label: {true_label}")
            print(f"Epsilon: {epsilon}")
            print(f"Timeout: {self.timeout}s per query")

        # Flatten image
        image_flat = image.squeeze().cpu().numpy().flatten()
        num_classes = len(self.network.outputVars[0])

        start_time = time.time()
        counterexample_class = None

        # Check each incorrect class
        for target_class in range(num_classes):
            if target_class == true_label:
                continue

            # Create fresh network for each query
            network_copy = Marabou.read_onnx(self.onnx_path)
            input_vars = network_copy.inputVars[0].flatten()  # Flatten to 1D
            output_vars = network_copy.outputVars[0].flatten()  # Flatten to 1D

            # Set input constraints
            for i, pixel_val in enumerate(image_flat):
                lower = max(pixel_bounds[0], pixel_val - epsilon)
                upper = min(pixel_bounds[1], pixel_val + epsilon)
                network_copy.setLowerBound(input_vars[i], float(lower))
                network_copy.setUpperBound(input_vars[i], float(upper))

            # Set output constraint: try to make target_class > true_label
            network_copy.addInequality(
                [output_vars[target_class], output_vars[true_label]],
                [1, -1],
                0.0001
            )

            if verbose:
                print(f"  Checking class {target_class}...", end=" ", flush=True)

            # Solve
            options = Marabou.createOptions(timeoutInSeconds=self.timeout)
            exit_code, vals, stats = network_copy.solve(options=options)

            if exit_code == "sat":
                if verbose:
                    print(f"⚠️  COUNTEREXAMPLE FOUND")
                counterexample_class = target_class
                break
            elif exit_code == "unsat":
                if verbose:
                    print(f"✓ Robust")
            else:
                if verbose:
                    print(f"? Inconclusive ({exit_code})")

        total_time = time.time() - start_time
        is_robust = (counterexample_class is None)

        if verbose:
            print(f"\n{'='*60}")
            if is_robust:
                print(f"✓ VERIFICATION SUCCESSFUL")
                print(f"  Model is locally robust within ε={epsilon}")
            else:
                print(f"✗ VERIFICATION FAILED")
                print(f"  Counterexample: class {counterexample_class}")
            print(f"  Total time: {total_time:.2f}s")
            print(f"{'='*60}")

        return is_robust, counterexample_class, total_time


def verify_local_robustness(onnx_path, test_image, true_label,
                            epsilon=0.02, timeout=60, verbose=True):
    """
    Convenience function for local robustness verification.

    Args:
        onnx_path: Path to ONNX model
        test_image: Input image tensor
        true_label: True class label
        epsilon: Perturbation bound
        timeout: Timeout per query in seconds
        verbose: Whether to print progress

    Returns:
        Tuple of (is_robust, counterexample_class, verification_time)
    """
    verifier = MarabouVerifier(onnx_path, timeout=timeout)
    return verifier.verify_local_robustness(
        test_image, true_label, epsilon, verbose=verbose
    )
