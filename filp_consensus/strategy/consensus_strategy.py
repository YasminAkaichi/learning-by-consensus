"""Flower strategy for Learning by Consensus.

Each client independently learns one local hypothesis H_i.

The server:
1. collects the local hypotheses;
2. keeps them separate;
3. constructs the ensemble {H_1, ..., H_K};
4. redistributes the ensemble to the clients;
5. collects Consensus evaluation metrics and prediction traces.

No rule merging, averaging, or weighting is performed.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np

from flwr.common import (
    EvaluateIns,
    FitIns,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.strategy import FedAvg


log = logging.getLogger(__name__)


def _encode_json(payload: Any) -> Parameters:
    """Serialize a JSON-compatible payload as Flower Parameters."""
    raw = json.dumps(
        payload,
        ensure_ascii=False,
    ).encode("utf-8")

    array = np.frombuffer(
        raw,
        dtype=np.uint8,
    ).copy()

    return ndarrays_to_parameters([array])


def _decode_json(parameters: Parameters) -> Any:
    """Deserialize Flower Parameters produced by _encode_json."""
    arrays = parameters_to_ndarrays(parameters)

    if not arrays or arrays[0].size == 0:
        return None

    raw = (
        arrays[0]
        .astype(np.uint8, copy=False)
        .tobytes()
    )

    return json.loads(
        raw.decode("utf-8")
    )


def _empty_parameters() -> Parameters:
    """Return initial empty Flower parameters."""
    return _encode_json(
        {
            "type": "empty",
        }
    )


class ConsensusStrategy(FedAvg):
    """Flower strategy implementing Learning by Consensus."""

    def __init__(
        self,
        *,
        num_clients: int,
        **kwargs,
    ) -> None:

        super().__init__(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=num_clients,
            min_evaluate_clients=num_clients,
            min_available_clients=num_clients,
            **kwargs,
        )

        self.num_clients = num_clients

        # Local hypotheses collected from clients.
        self.hypotheses: list[
            dict[str, Any]
        ] = []

        # Final Consensus evaluation metrics.
        self.consensus_metrics: dict[
            str,
            Any,
        ] = {}

        # Detailed prediction/voting information.
        self.vote_traces: list[
            dict[str, Any]
        ] = []

        # Cost measurements (one entry per client / per direction).
        self.induction_times: dict[str, float] = {}
        self.upload_bytes: dict[str, int] = {}
        self.ensemble_bytes: int = 0

        # client_id -> [{hypothesis, tp, fn, tn, fp}, ...]: coverage of each
        # hypothesis on that client's own training data.
        self.local_coverage: dict[str, list[dict[str, Any]]] = {}

    def initialize_parameters(
        self,
        client_manager: ClientManager,
    ) -> Parameters:
        """Initial parameters contain no hypothesis ensemble yet."""

        return _empty_parameters()

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ):
        """Ask every client to learn one local hypothesis."""

        clients = client_manager.sample(
            num_clients=self.num_clients,
            min_num_clients=self.num_clients,
        )

        fit_ins = FitIns(
            parameters,
            {
                "server_round": server_round,
            },
        )

        return [
            (client, fit_ins)
            for client in clients
        ]

    def aggregate_fit(
        self,
        server_round,
        results,
        failures,
    ):
        """Collect H_i and construct the hypothesis ensemble."""

        if failures:
            log.warning(
                "Round %s received %d client failures.",
                server_round,
                len(failures),
            )

        hypotheses: list[
            dict[str, Any]
        ] = []

        for client_proxy, fit_res in results:

            payload = _decode_json(
                fit_res.parameters
            )

            if not payload:
                log.warning(
                    "Client %s returned an empty payload.",
                    client_proxy.cid,
                )
                continue

            if (
                payload.get("type")
                != "local_hypothesis"
            ):
                log.warning(
                    "Ignoring unexpected payload "
                    "from client %s: %s",
                    client_proxy.cid,
                    payload.get("type"),
                )
                continue

            hypothesis = payload.get(
                "hypothesis",
                [],
            )

            client_key = str(
                payload.get("client_id", client_proxy.cid)
            )
            self.induction_times[client_key] = float(
                fit_res.metrics.get("induction_time", 0.0)
            )
            self.upload_bytes[client_key] = sum(
                len(tensor)
                for tensor in fit_res.parameters.tensors
            )

            if not hypothesis:
                log.warning(
                    "Client %s learned an empty hypothesis.",
                    payload.get(
                        "client_id",
                        client_proxy.cid,
                    ),
                )

            # IMPORTANT:
            # Empty hypotheses are kept.
            # Every client remains one voter.
            hypotheses.append(
                {
                    "client_id": payload.get(
                        "client_id",
                        client_proxy.cid,
                    ),
                    "learner": payload.get(
                        "learner"
                    ),
                    "hypothesis": hypothesis,
                }
            )

        # Stable ordering:
        # H1, H2, ..., Hk
        hypotheses.sort(
            key=lambda item: str(
                item["client_id"]
            )
        )

        self.hypotheses = hypotheses

        log.info(
            "Consensus round %s collected %d hypotheses.",
            server_round,
            len(self.hypotheses),
        )

        for item in self.hypotheses:
            log.info(
                "H%s [%s] -> %d rules",
                item["client_id"],
                item["learner"],
                len(item["hypothesis"]),
            )

        ensemble = {
            "type": "hypothesis_ensemble",
            "server_round": server_round,
            "hypotheses": self.hypotheses,
        }

        # This is NOT a merged symbolic hypothesis.
        # It is the ensemble [H1, ..., Hk].
        parameters_aggregated = _encode_json(
            ensemble
        )

        # Size of the ensemble sent back to EACH client.
        self.ensemble_bytes = sum(
            len(tensor)
            for tensor in parameters_aggregated.tensors
        )

        metrics = {
            "n_hypotheses": len(
                self.hypotheses
            ),
        }

        return (
            parameters_aggregated,
            metrics,
        )

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ):
        """Redistribute the complete ensemble to every client."""

        if not self.hypotheses:
            return []

        clients = client_manager.sample(
            num_clients=self.num_clients,
            min_num_clients=self.num_clients,
        )

        evaluate_ins = EvaluateIns(
            parameters,
            {
                "server_round": server_round,
            },
        )

        return [
            (client, evaluate_ins)
            for client in clients
        ]

    def aggregate_evaluate(
        self,
        server_round,
        results,
        failures,
    ):
        """Aggregate Consensus evaluation results.

        Every client evaluates the same hypothesis ensemble
        on the same global test set.

        Therefore:
        - Consensus metrics should be identical;
        - prediction traces should also be identical.

        One client result is kept as the canonical result.
        """

        if failures:
            log.warning(
                "Consensus evaluation round %s "
                "received %d failures.",
                server_round,
                len(failures),
            )

        if not results:
            log.warning(
                "Consensus evaluation round %s "
                "returned no results.",
                server_round,
            )

            return None, {}

        metrics_list = [
            evaluate_res.metrics
            for _, evaluate_res in results
        ]

        # --------------------------------------------------------
        # 1. Consensus metrics
        # --------------------------------------------------------

        consensus_keys = (
            "tp",
            "tn",
            "fp",
            "fn",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "mcc",
            "n_hypotheses",
        )

        reference_metrics = {
            key: metrics_list[0][key]
            for key in consensus_keys
        }

        # --------------------------------------------------------
        # 2. Verify that all clients obtained the same metrics
        # --------------------------------------------------------

        for metrics in metrics_list[1:]:

            comparable_metrics = {
                key: metrics[key]
                for key in consensus_keys
            }

            if (
                comparable_metrics
                != reference_metrics
            ):
                log.warning(
                    "Consensus evaluation metrics "
                    "differ between clients: "
                    "%s != %s",
                    reference_metrics,
                    comparable_metrics,
                )

        # Local coverage of every hypothesis on each client's own data.
        for client_proxy, evaluate_res in results:
            metrics = evaluate_res.metrics
            try:
                self.local_coverage[
                    str(metrics.get("client_id", client_proxy.cid))
                ] = json.loads(metrics.get("local_coverage", "[]"))
            except (TypeError, json.JSONDecodeError):
                pass

        # --------------------------------------------------------
        # 3. Recover detailed prediction traces
        # --------------------------------------------------------

        raw_vote_traces = metrics_list[
            0
        ].get(
            "vote_traces",
            "[]",
        )

        try:
            self.vote_traces = json.loads(
                raw_vote_traces
            )

        except (
            TypeError,
            json.JSONDecodeError,
        ):
            log.warning(
                "Could not decode Consensus "
                "vote traces."
            )

            self.vote_traces = []

        log.info(
            "Stored %d Consensus vote traces.",
            len(self.vote_traces),
        )

        # --------------------------------------------------------
        # 4. Store final Consensus metrics
        # --------------------------------------------------------

        self.consensus_metrics = {
            **reference_metrics,
            "learner": metrics_list[
                0
            ].get("learner"),
        }

        log.info(
            "Consensus metrics: %s",
            self.consensus_metrics,
        )

        # --------------------------------------------------------
        # 5. Return Flower evaluation result
        # --------------------------------------------------------

        loss = results[0][1].loss

        return (
            loss,
            self.consensus_metrics,
        )