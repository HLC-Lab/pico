# Copyright (c) 2025 Daniele De Sensi e Saverio Pasqualoni
# Licensed under the MIT License

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Iterable


def _empty_data_dict() -> Dict[str, list]:
    return {
        "Message": [],
        "message_bytes": [],
        "latency": [],
        "bandwidth": [],
        "Cluster": [],
        "collective": [],
        "iteration": [],
        "run": [],
    }


@dataclass
class RefinedDataset:
    data: Dict[str, list] = field(default_factory=_empty_data_dict)

    def reset(self) -> None:
        for key in self.data:
            self.data[key].clear()


def load_data(
    dataset: RefinedDataset,
    cluster: str,
    nodes: int,
    path: str,
    messages: Iterable[str],
    *,
    coll: str | None = None,
    congested: bool = False,
) -> RefinedDataset:
    for msg in messages:
        msg_value, msg_unit = msg.strip().split(" ")
        multiplier = {
            "B": 1,
            "KiB": 1024,
            "MiB": 1024**2,
            "GiB": 1024**3,
        }.get(msg_unit)
        if multiplier is None:
            raise ValueError(f"Unknown message size unit in {msg}")

        message_bytes = int(msg_value) * multiplier
        if not os.path.isdir(path):
            continue

        for file_name in os.listdir(path):
            if not congested and len(file_name.strip().split("_")) == 4:
                continue
            if congested and len(file_name.strip().split("_")) == 3:
                continue

            parts = file_name.strip().split("_")
            found_message_bytes = int(parts[0])
            if found_message_bytes != message_bytes:
                continue

            collective = parts[1].split(".")[0]
            if coll is not None and collective != coll:
                continue

            file_path = os.path.join(path, file_name)
            iterations = []
            latencies = []
            with open(file_path, "r") as file:
                lines = file.readlines()[2:]
                for idx, line in enumerate(lines, start=1):
                    latencies.append(float(line.strip()))
                    iterations.append(idx)

            if collective == "all2all":
                gb_sent = ((message_bytes / 1e9) * (nodes - 1)) * 8
            elif collective in {"allgather", "reducescatter"}:
                gb_sent = (message_bytes / 1e9) * ((nodes - 1) / nodes) * 8
            elif collective == "allreduce":
                gb_sent = 2 * (message_bytes / 1e9) * ((nodes - 1) / nodes) * 8
            elif collective == "pointpoint":
                gb_sent = (message_bytes / 1e9) * 8
            else:
                gb_sent = 0

            bandwidth = [gb_sent / x for x in latencies if x != 0]

            dataset.data["latency"].extend(latencies)
            dataset.data["iteration"].extend(iterations)
            dataset.data["bandwidth"].extend(bandwidth)
            dataset.data["Message"].extend([msg] * len(latencies))
            dataset.data["message_bytes"].extend([message_bytes] * len(latencies))
            dataset.data["Cluster"].extend([cluster] * len(latencies))
            dataset.data["collective"].extend([collective] * len(latencies))
            dataset.data["run"].extend([file_name] * len(latencies))
    return dataset
