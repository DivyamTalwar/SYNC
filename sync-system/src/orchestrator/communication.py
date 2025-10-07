from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
from src.core.agent import Message
from src.utils.logging import get_logger

logger = get_logger("communication")


@dataclass
class DialogueTurn:
    round_number: int
    messages: List[Message] = field(default_factory=list)
    timestamp: Optional[float] = None

    def get_messages_by_sender(self, sender_id: int) -> List[Message]:
        return [m for m in self.messages if m.sender_id == sender_id]

    def get_messages_to_receiver(self, receiver_id: int) -> List[Message]:
        return [m for m in self.messages if m.receiver_id == receiver_id or m.receiver_id == -1]


class CommunicationChannel:

    def __init__(self, num_agents: int):
        self.num_agents = num_agents
        self.current_round = 0

        self.dialogue_turns: List[DialogueTurn] = []
        self.all_messages: List[Message] = []

        self.message_queues: Dict[int, List[Message]] = {
            i: [] for i in range(num_agents)
        }

        # Statistics
        self.total_messages = 0
        self.broadcast_count = 0
        self.targeted_count = 0

        logger.info(f"Initialized CommunicationChannel for {num_agents} agents")

    def send_message(self, message: Message) -> None:
        self.all_messages.append(message)
        self.total_messages += 1

        if message.receiver_id == -1:
            self.broadcast_count += 1
            for agent_id in range(self.num_agents):
                if agent_id != message.sender_id:
                    self.message_queues[agent_id].append(message)

            logger.debug(
                f"Broadcast message from Agent {message.sender_id} "
                f"to {self.num_agents - 1} agents"
            )
        else:
            self.targeted_count += 1
            if 0 <= message.receiver_id < self.num_agents:
                self.message_queues[message.receiver_id].append(message)
                logger.debug(
                    f"Sent targeted message from Agent {message.sender_id} "
                    f"to Agent {message.receiver_id}"
                )
            else:
                logger.warning(f"Invalid receiver_id: {message.receiver_id}")

    def get_messages_for_agent(self, agent_id: int, clear: bool = True) -> List[Message]:
        messages = self.message_queues[agent_id].copy()

        if clear:
            self.message_queues[agent_id].clear()

        logger.debug(f"Retrieved {len(messages)} messages for Agent {agent_id}")
        return messages

    def broadcast_messages(self, messages: List[Message]) -> None:
        for message in messages:
            self.send_message(message)

    def complete_round(self, round_number: Optional[int] = None) -> DialogueTurn:
        if round_number is None:
            round_number = self.current_round

        round_messages = [m for m in self.all_messages if m.step == round_number]

        turn = DialogueTurn(
            round_number=round_number,
            messages=round_messages,
        )

        self.dialogue_turns.append(turn)
        self.current_round += 1

        logger.info(
            f"Completed round {round_number}: "
            f"{len(round_messages)} messages, "
            f"{self.broadcast_count} broadcasts, "
            f"{self.targeted_count} targeted"
        )

        return turn

    def get_dialogue_history(self, max_turns: Optional[int] = None) -> List[DialogueTurn]:
        if max_turns is None:
            return self.dialogue_turns
        else:
            return self.dialogue_turns[-max_turns:]

    def get_messages_from_agent(self, agent_id: int) -> List[Message]:
        return [m for m in self.all_messages if m.sender_id == agent_id]

    def get_messages_to_agent(self, agent_id: int) -> List[Message]:
        return [
            m for m in self.all_messages
            if m.receiver_id == agent_id or m.receiver_id == -1
        ]

    def get_conversation_between(
        self,
        agent1_id: int,
        agent2_id: int
    ) -> List[Message]:
        return [
            m for m in self.all_messages
            if (m.sender_id == agent1_id and (m.receiver_id == agent2_id or m.receiver_id == -1))
            or (m.sender_id == agent2_id and (m.receiver_id == agent1_id or m.receiver_id == -1))
        ]

    def get_statistics(self) -> Dict:
        return {
            "total_messages": self.total_messages,
            "broadcast_messages": self.broadcast_count,
            "targeted_messages": self.targeted_count,
            "total_rounds": len(self.dialogue_turns),
            "current_round": self.current_round,
            "messages_per_round": (
                self.total_messages / len(self.dialogue_turns)
                if self.dialogue_turns else 0
            ),
        }

    def reset(self) -> None:
        self.current_round = 0
        self.dialogue_turns.clear()
        self.all_messages.clear()

        for agent_id in range(self.num_agents):
            self.message_queues[agent_id].clear()

        self.total_messages = 0
        self.broadcast_count = 0
        self.targeted_count = 0

        logger.info("Reset communication channel")

    def format_dialogue_history(self, max_chars_per_message: int = 100) -> str:
        lines = ["=== DIALOGUE HISTORY ===\n"]

        for turn in self.dialogue_turns:
            lines.append(f"\n--- Round {turn.round_number} ---")

            for msg in turn.messages:
                sender = f"Agent {msg.sender_id}"
                receiver = "ALL" if msg.receiver_id == -1 else f"Agent {msg.receiver_id}"
                content = msg.content[:max_chars_per_message]
                if len(msg.content) > max_chars_per_message:
                    content += "..."

                lines.append(f"{sender} -> {receiver}: {content}")
                lines.append(f"  [Objective: {msg.objective}]")

        return "\n".join(lines)


class MessageFilter:

    @staticmethod
    def filter_by_objective(
        messages: List[Message],
        objectives: List[str]
    ) -> List[Message]:
        return [m for m in messages if m.objective in objectives]

    @staticmethod
    def filter_by_sender(
        messages: List[Message],
        sender_ids: List[int]
    ) -> List[Message]:
        return [m for m in messages if m.sender_id in sender_ids]

    @staticmethod
    def filter_by_receiver(
        messages: List[Message],
        receiver_ids: List[int]
    ) -> List[Message]:
        return [
            m for m in messages
            if m.receiver_id in receiver_ids or m.receiver_id == -1
        ]

    @staticmethod
    def filter_by_round(
        messages: List[Message],
        round_numbers: List[int]
    ) -> List[Message]:
        return [m for m in messages if m.step in round_numbers]

    @staticmethod
    def get_recent_messages(
        messages: List[Message],
        n: int
    ) -> List[Message]:
        return messages[-n:] if len(messages) > n else messages
