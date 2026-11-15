"""
aadc_extensions.py — VectorFunctionSelectiveAD wrapper over aadc.Functions.

Batch evaluate with AVX + multithreading. Supports:
  evaluate()          — full reverse over all outputs, returns jac_z AND jac_theta
  evaluate_selective() — reverse only for selected outputs
  evaluate_selective_theta() — same, adjoints w.r.t. z_args AND param_args
  evaluate_forward()   — forward only (no reverse)
"""
from typing import Any, Dict, List, Optional
import numpy as np
from aadc import Functions, evaluate, Argument
import os
import platform
try:
    from aadc import evaluate_core as _evaluate_core      # aadc >= 2.x: lets the vector width be chosen
    from aadc import ErrorBuffer as _ErrorBuffer
except ImportError:
    _evaluate_core = _ErrorBuffer = None
try:
    from concurrent.futures import ThreadPoolExecutor as ThreadPool
except ImportError:
    pass

try:
    from aadc import ErrorCollectionMode
except ImportError:
    class ErrorCollectionMode:
        NONE = None


def _best_vector_kind():
    """'avx512' on x86-64 Linux CPUs with AVX-512F, else None (engine default)."""
    try:
        if platform.machine().lower() in ("x86_64", "amd64") and os.path.exists("/proc/cpuinfo"):
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("flags"):
                        return "avx512" if " avx512f" in line else None
    except OSError:
        pass
    return None


class VectorFunctionSelectiveAD:

    def __init__(self, funcs: Functions, args: Any, res: Any, *,
                 param_args: Any = (), batch_param_args: Any = (),
                 num_threads: int = 1, vector_kind: Any = "best") -> None:
        """vector_kind: AADC vector workspace for the batched replays: "best" (AVX-512 where the CPU has it,
        else the engine default), None / "auto" (engine default), or "scalar" / "avx2" / "avx512".
        Results are bitwise identical across kinds; only speed changes."""
        self._funcs = funcs
        if vector_kind == "best":
            vector_kind = _best_vector_kind()
        self._vector_kind = None if vector_kind in (None, "auto") or _evaluate_core is None else vector_kind
        self._args = list(np.ravel(np.asarray(args, dtype=object)))
        self._res = list(np.ravel(np.asarray(res, dtype=object)))
        self._param_args = list(np.ravel(np.asarray(param_args, dtype=object))) \
            if len(np.atleast_1d(param_args)) else []
        self._batch_param_args = list(
            np.ravel(np.asarray(batch_param_args, dtype=object))) \
            if len(np.atleast_1d(batch_param_args)) else []
        self._num_threads = num_threads
        self._params: Dict[Argument, float] = {}
        self._batch_params: Dict[Argument, Any] = {}
        self._fixed: Dict[Argument, float] = {}

    def _evaluate(self, funcs, request, inputs, workers, error_mode=ErrorCollectionMode.NONE):
        """aadc.evaluate, on the chosen vector workspace when no error collection is requested."""
        if self._vector_kind is not None and (error_mode is None or error_mode == ErrorCollectionMode.NONE):
            try:
                values, derivs = _evaluate_core(funcs, request, inputs, int(workers or 1), self._vector_kind)
                return values, derivs, _ErrorBuffer()
            except ValueError:            # tape with a scalar-only block: fall back to the engine default
                self._vector_kind = None
        return evaluate(funcs=funcs, request=request, inputs=inputs, workers=workers, error_mode=error_mode)

    def set_params(self, params: Any) -> None:
        params = np.atleast_1d(np.asarray(params, dtype=np.float64)).ravel()
        if len(params) != len(self._param_args):
            raise ValueError("set_params: wrong number of parameters")
        self._params = {a: float(p) for a, p in zip(self._param_args, params)}

    def set_fixed_inputs(self, fixed: Dict[Argument, float]) -> None:
        """Constant inputs that are neither args nor param_args (no adjoints).

        aadc.evaluate does not reject a missing input: it reads whatever is in
        the workspace (garbage, or a stale value from a previous call), so every
        recorded input must be passed.
        """
        self._fixed = {a: float(v) for a, v in fixed.items()}

    def set_batch_params(self, batch_params: Any) -> None:
        bp = np.asarray(batch_params, dtype=np.float64)
        bp2 = bp.reshape(bp.shape[0], -1) if bp.ndim > 1 else bp.reshape(-1, 1)
        if bp2.shape[1] != len(self._batch_param_args):
            raise ValueError("set_batch_params: wrong number of parameters")
        self._batch_params = {
            a: np.ascontiguousarray(bp2[:, j])
            for j, a in enumerate(self._batch_param_args)
        }

    def evaluate(self, x: Any,
                 error_mode: "ErrorCollectionMode" = ErrorCollectionMode.NONE):
        """Full reverse over all outputs.

        Returns (values, jac_z, jac_theta, errors) where:
          values:    (M, n_out)
          jac_z:     (M, n_out, d)       — derivatives w.r.t. z_args
          jac_theta: (M, n_out, n_theta) — derivatives w.r.t. param_args
          errors:    error info from aadc
        """
        x = np.asarray(x, dtype=np.float64)
        squeeze = x.ndim == 1
        x2 = x.reshape(1, -1) if squeeze else x
        if x2.shape[1] != len(self._args):
            raise ValueError("evaluate: wrong input dimension")
        inputs: Dict[Argument, Any] = {
            a: np.ascontiguousarray(x2[:, j]) for j, a in enumerate(self._args)
        }
        inputs.update(self._params)
        inputs.update(self._fixed)
        inputs.update(self._batch_params)
        # Request adjoints for BOTH z_args and param_args
        all_diff_args = self._args + self._param_args
        request = {r: all_diff_args for r in self._res}
        values, derivs, errors = self._evaluate(
            funcs=self._funcs, request=request, inputs=inputs,
            workers=self._num_threads, error_mode=error_mode)
        batch = x2.shape[0]
        for bp in self._batch_params.values():
            batch = max(batch, len(np.atleast_1d(bp)))
        out = np.empty((batch, len(self._res)))
        jac_z = np.empty((batch, len(self._res), len(self._args)))
        jac_theta = np.empty((batch, len(self._res), len(self._param_args))) \
            if self._param_args else None
        for k, r in enumerate(self._res):
            out[:, k] = values[r]
            for j, a in enumerate(self._args):
                jac_z[:, k, j] = derivs[r][a]
            if jac_theta is not None:
                for j, a in enumerate(self._param_args):
                    jac_theta[:, k, j] = derivs[r][a]
        return out, jac_z, jac_theta, errors

    def evaluate_selective(self, x: Any, outputs: List,
                           error_mode: "ErrorCollectionMode" = ErrorCollectionMode.NONE):
        """Reverse only for selected outputs. Values returned for ALL outputs."""
        x = np.asarray(x, dtype=np.float64)
        x2 = x.reshape(1, -1) if x.ndim == 1 else x
        inputs: Dict[Argument, Any] = {
            a: np.ascontiguousarray(x2[:, j]) for j, a in enumerate(self._args)
        }
        inputs.update(self._params)
        inputs.update(self._fixed)
        inputs.update(self._batch_params)
        # Request: selected outputs get full reverse, others get values only
        request = {}
        for r in self._res:
            request[r] = self._args if r in outputs else []
        values_dict, derivs, errors = self._evaluate(
            funcs=self._funcs, request=request, inputs=inputs,
            workers=self._num_threads, error_mode=error_mode)
        batch = x2.shape[0]
        out = np.empty((batch, len(self._res)))
        for k, r in enumerate(self._res):
            out[:, k] = values_dict[r]
        jac = np.empty((batch, len(outputs), len(self._args)))
        for k, r in enumerate(outputs):
            for j, a in enumerate(self._args):
                jac[:, k, j] = derivs[r][a]
        return out, jac, errors

    def evaluate_selective_theta(self, x: Any, outputs: List,
                                 error_mode: "ErrorCollectionMode" = ErrorCollectionMode.NONE):
        """Reverse only for selected outputs, adjoints w.r.t. z_args AND param_args.

        Same reverse passes as evaluate_selective (one per selected output);
        the parameter adjoints come from the same sweep.
        Returns (values, jac_z, jac_theta, errors):
          values:    (M, n_out)              — all outputs
          jac_z:     (M, len(outputs), d)
          jac_theta: (M, len(outputs), n_theta) or None if no param_args
        """
        x = np.asarray(x, dtype=np.float64)
        x2 = x.reshape(1, -1) if x.ndim == 1 else x
        inputs: Dict[Argument, Any] = {
            a: np.ascontiguousarray(x2[:, j]) for j, a in enumerate(self._args)
        }
        inputs.update(self._params)
        inputs.update(self._fixed)
        inputs.update(self._batch_params)
        diff_args = self._args + self._param_args
        request = {r: (diff_args if r in outputs else []) for r in self._res}
        values_dict, derivs, errors = self._evaluate(
            funcs=self._funcs, request=request, inputs=inputs,
            workers=self._num_threads, error_mode=error_mode)
        batch = x2.shape[0]
        out = np.empty((batch, len(self._res)))
        for k, r in enumerate(self._res):
            out[:, k] = values_dict[r]
        jac = np.empty((batch, len(outputs), len(self._args)))
        jac_theta = np.empty((batch, len(outputs), len(self._param_args))) \
            if self._param_args else None
        for k, r in enumerate(outputs):
            for j, a in enumerate(self._args):
                jac[:, k, j] = derivs[r][a]
            if jac_theta is not None:
                for j, a in enumerate(self._param_args):
                    jac_theta[:, k, j] = derivs[r][a]
        return out, jac, jac_theta, errors

    def _inputs_from_columns(self, cols):
        inputs: Dict[Argument, Any] = {a: c for a, c in zip(self._args, cols)}
        inputs.update(self._params)
        inputs.update(self._fixed)
        inputs.update(self._batch_params)
        return inputs

    def newton_step_columns(self, cols: List, output, direction: Any,
                            error_mode: "ErrorCollectionMode" = ErrorCollectionMode.NONE):
        """One reverse pass of a single output on inputs given as contiguous columns.

        Returns (g, dg_dir, dg_dtheta): the output's values (n,), the directional derivative
        sum_j dg/dz_j * direction_j (n,) (zero entries of `direction` are skipped), and dg/dtheta
        (n, n_theta) or None. No (n, d) Jacobian is built and no input column is copied.
        """
        inputs = self._inputs_from_columns(cols)
        request = {r: ((self._args + self._param_args) if r in (output,) else []) for r in self._res}
        values_dict, derivs, errors = self._evaluate(
            funcs=self._funcs, request=request, inputs=inputs,
            workers=self._num_threads, error_mode=error_mode)
        dr = derivs[output]
        dg_dir = np.zeros(len(cols[0]))
        for a, vj in zip(self._args, direction):
            if vj != 0.0:
                dg_dir += dr[a] * vj
        dg_dt = np.column_stack([dr[a] for a in self._param_args]) if self._param_args else None
        return np.asarray(values_dict[output]), dg_dir, dg_dt

    def forward_columns(self, cols: List, output,
                        error_mode: "ErrorCollectionMode" = ErrorCollectionMode.NONE):
        """Forward only on inputs given as contiguous columns; values of one output (n,)."""
        inputs = self._inputs_from_columns(cols)
        values_dict, _, _ = self._evaluate(funcs=self._funcs, request={r: [] for r in self._res},
                                     inputs=inputs, workers=self._num_threads, error_mode=error_mode)
        return np.asarray(values_dict[output])

    def evaluate_forward(self, x: Any,
                         error_mode: "ErrorCollectionMode" = ErrorCollectionMode.NONE):
        """Forward only — no reverse pass."""
        x = np.asarray(x, dtype=np.float64)
        x2 = x.reshape(1, -1) if x.ndim == 1 else x
        inputs: Dict[Argument, Any] = {
            a: np.ascontiguousarray(x2[:, j]) for j, a in enumerate(self._args)
        }
        inputs.update(self._params)
        inputs.update(self._fixed)
        inputs.update(self._batch_params)
        request = {r: [] for r in self._res}
        values_dict, _, errors = self._evaluate(
            funcs=self._funcs, request=request, inputs=inputs,
            workers=self._num_threads, error_mode=error_mode)
        batch = x2.shape[0]
        out = np.empty((batch, len(self._res)))
        for k, r in enumerate(self._res):
            out[:, k] = values_dict[r]
        return out
