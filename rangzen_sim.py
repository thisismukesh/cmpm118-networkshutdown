import random
import math
from dataclasses import dataclass, field
from typing import Set, Dict, List

import numpy as np
import matplotlib.pyplot as plt

@dataclass
class Node:
    node_id: int
    friends: Set[int]
    messages: Dict[int, float] = field(default_factory=dict)

    def has_message(self, msg_id: int) -> bool:
        return msg_id in self.messages

    def receive_message(self, msg_id: int, incoming_priority: float, trust_value: float, noise_std: float = 0.05):
        """
        Update or store a message with a new priority based on trust + noise.
        """
        # Simple trust-based scaling (you can tune this later)
        # Map trust_value (0..max) into [0,1] approximately, then apply
        trust_factor = 1.0 / (1.0 + math.exp(-trust_value))  # sigmoid

        base_priority = incoming_priority * trust_factor

        # Add small Gaussian noise
        noise = np.random.normal(0.0, noise_std)
        new_priority = base_priority + noise

        # Clamp to [0,1]
        new_priority = max(0.0, min(1.0, new_priority))

        # Store max priority seen so far for this message
        if msg_id not in self.messages or new_priority > self.messages[msg_id]:
            self.messages[msg_id] = new_priority

def create_nodes(num_nodes: int, avg_friends: int = 10, seed: int = 42) -> List[Node]:
    random.seed(seed)
    np.random.seed(seed)

    # Generate friend sets: for simplicity, each node picks some random friends
    nodes = []
    for i in range(num_nodes):
        # choose a random number of friends around avg_friends
        num_friends = max(1, int(random.gauss(avg_friends, avg_friends / 3)))
        possible_friends = [j for j in range(num_nodes) if j != i]
        friends = set(random.sample(possible_friends, k=min(num_friends, len(possible_friends))))
        nodes.append(Node(node_id=i, friends=friends))
    return nodes

def compute_trust_matrix(nodes: List[Node]) -> np.ndarray:
    """
    trust_matrix[i, j] = number of mutual friends between node i and node j
    """
    n = len(nodes)
    trust_matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        fi = nodes[i].friends
        for j in range(i + 1, n):
            fj = nodes[j].friends
            mutual = len(fi.intersection(fj))
            trust_matrix[i, j] = mutual
            trust_matrix[j, i] = mutual
    return trust_matrix

def simulate(
    mode_name: str,
    num_nodes: int = 100,
    timesteps: int = 50,
    contacts_per_step: int = 150,
    p_discovery: float = 1.0,
    p_connection: float = 0.9,
    p_sleep: float = 0.0,
    noise_std: float = 0.05,
) -> List[int]:
    """
    Simulate message propagation in a simple encounter-based model.
    Returns: list of coverage counts over time.
    """
    # Setup nodes + trust
    nodes = create_nodes(num_nodes=num_nodes, avg_friends=12)
    trust_matrix = compute_trust_matrix(nodes)

    # Initialize: message 0 starts at node 0 with priority 1.0
    MSG_ID = 0
    nodes[0].messages[MSG_ID] = 1.0

    coverage_over_time: List[int] = []

    for t in range(timesteps):
        # Determine which nodes are "sleeping" (e.g., due to Doze)
        sleeping = set()
        if p_sleep > 0.0:
            for i in range(num_nodes):
                if random.random() < p_sleep:
                    sleeping.add(i)

        # Generate random encounters
        for _ in range(contacts_per_step):
            a = random.randrange(num_nodes)
            b = random.randrange(num_nodes)
            if a == b:
                continue
            if a in sleeping or b in sleeping:
                continue

            # Discovery step
            if random.random() > p_discovery:
                continue

            # Connection step
            if random.random() > p_connection:
                continue

            # At this point, we have a successful "contact" where message exchange can occur
            trust_ab = trust_matrix[a, b]
            trust_ba = trust_ab  # symmetric in our simple model

            node_a = nodes[a]
            node_b = nodes[b]

            # A -> B
            if node_a.has_message(MSG_ID) and not node_b.has_message(MSG_ID):
                incoming_priority = node_a.messages[MSG_ID]
                node_b.receive_message(MSG_ID, incoming_priority, trust_ab, noise_std=noise_std)

            # B -> A
            if node_b.has_message(MSG_ID) and not node_a.has_message(MSG_ID):
                incoming_priority = node_b.messages[MSG_ID]
                node_a.receive_message(MSG_ID, incoming_priority, trust_ba, noise_std=noise_std)

        # After all encounters, record coverage
        coverage = sum(1 for n in nodes if n.has_message(MSG_ID))
        coverage_over_time.append(coverage)

        # Optional: print a bit of progress
        # print(f"[{mode_name}] t={t}, coverage={coverage}/{num_nodes}")

    return coverage_over_time

def main():
    num_nodes = 100
    timesteps = 50
    contacts_per_step = 150

    # Ideal 2015-like mode
    ideal_coverage = simulate(
        mode_name="ideal",
        num_nodes=num_nodes,
        timesteps=timesteps,
        contacts_per_step=contacts_per_step,
        p_discovery=1.0,
        p_connection=0.9,
        p_sleep=0.0,
        noise_std=0.05,
    )

    # Modern Android constrained mode (tune these if you want)
    modern_coverage = simulate(
        mode_name="modern",
        num_nodes=num_nodes,
        timesteps=timesteps,
        contacts_per_step=contacts_per_step,
        p_discovery=0.4,   # harder to discover peers
        p_connection=0.6,  # more BT failures
        p_sleep=0.3,       # 30% of nodes "sleeping" each round
        noise_std=0.05,
    )

    # Plot
    steps = list(range(timesteps))
    plt.figure()
    plt.plot(steps, [c / num_nodes for c in ideal_coverage], label="Ideal (old Android)")
    plt.plot(steps, [c / num_nodes for c in modern_coverage], label="Modern Android constraints")
    plt.xlabel("Timestep")
    plt.ylabel("Fraction of nodes that have seen the message")
    plt.title("Rangzen-like message propagation under ideal vs constrained conditions")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()