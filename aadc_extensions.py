"""
aadc_extensions.py — VectorFunctionSelectiveAD wrapper over aadc.Functions.

Batch evaluate with AVX + multithreading. Supports:
  evaluate()          — full reverse over all outputs, returns jac_z AND jac_theta
  evaluate_selective() — reverse only for selected outputs
  evaluate_forward()   — forward only (no reverse)
"""
from typing import Any, Dict, List, Optional
import numpy as np
from aadc import Functions, evaluate, Argument
try:
    from concurrent.futures import ThreadPoolExecutor as ThreadPool
except ImportError:
    pass

try:
    from aadc import ErrorCollectionMode
except ImportError:
    class ErrorCollectionMode:
        NONE = None


class VectorFunctionSelectiveAD:

    def __init__(self, funcs: Functions, args: Any, res: Any, *,
                 param_args: Any = (), batch_param_args: Any = (),
                 num_threads: int = 1) -> None:
        self._funcs = funcs
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

    def set_params(self, params: Any) -> None:
        params = np.atleast_1d(np.asarray(params, dtype=np.float64)).ravel()
        if len(params) != len(self._param_args):
            raise ValueError("set_params: wrong number of parameters")
        self._params = {a: float(p) for a, p in zip(self._param_args, params)}

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
        inputs.update(self._batch_params)
        # Request adjoints for BOTH z_args and param_args
        all_diff_args = self._args + self._param_args
        request = {r: all_diff_args for r in self._res}
        values, derivs, errors = evaluate(
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
        inputs.update(self._batch_params)
        # Request: selected outputs get full reverse, others get values only
        request = {}
        for r in self._res:
            request[r] = self._args if r in outputs else []
        values_dict, derivs, errors = evaluate(
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

    def evaluate_forward(self, x: Any,
                         error_mode: "ErrorCollectionMode" = ErrorCollectionMode.NONE):
        """Forward only — no reverse pass."""
        x = np.asarray(x, dtype=np.float64)
        x2 = x.reshape(1, -1) if x.ndim == 1 else x
        inputs: Dict[Argument, Any] = {
            a: np.ascontiguousarray(x2[:, j]) for j, a in enumerate(self._args)
        }
        inputs.update(self._params)
        inputs.update(self._batch_params)
        request = {r: [] for r in self._res}
        values_dict, _, errors = evaluate(
            funcs=self._funcs, request=request, inputs=inputs,
            workers=self._num_threads, error_mode=error_mode)
        batch = x2.shape[0]
        out = np.empty((batch, len(self._res)))
        for k, r in enumerate(self._res):
            out[:, k] = values_dict[r]
        return out
