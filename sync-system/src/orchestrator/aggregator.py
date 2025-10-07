from typing import List, Dict, Optional
from dataclasses import dataclass
from src.core.agent import Message
from src.core.state import AgentState
from src.llm.openrouter import OpenRouterClient
from src.utils.logging import get_logger
from config.base import config

logger = get_logger("aggregator")


@dataclass
class AgregatedResponse:
    final_answer: str
    confidence: float
    consensus_level: float 
    reasoning_summary: str
    conflicts: List[str]
    agent_contributions: Dict[int, str]
    metadata: Dict


class ResponseAggregator:

    def __init__(
        self,
        llm_client: Optional[OpenRouterClient] = None,
        use_aggregator_model: bool = True,
    ):
        if llm_client is None:
            self.llm_client = OpenRouterClient()
        else:
            self.llm_client = llm_client

        self.model = config.api.aggregator_model if use_aggregator_model else config.api.primary_model

        logger.info(f"Initialized ResponseAggregator with model: {self.model}")

    async def aggregate(
        self,
        query: str,
        agent_states: List[AgentState],
        messages: List[Message],
        dialogue_history: str,
    ) -> AgregatedResponse:
        prompt = self._build_aggregation_prompt(
            query=query,
            agent_states=agent_states,
            messages=messages,
            dialogue_history=dialogue_history,
        )

        messages_for_llm = [
            {
                "role": "system",
                "content": self._get_system_prompt()
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        try:
            response = await self.llm_client.generate_async(
                messages=messages_for_llm,
                model=self.model,
                temperature=0.3, 
                max_tokens=2048,
            )

            response_text = self.llm_client.extract_content(response)

            aggregated = self._parse_response(response_text, agent_states)

            logger.info(
                f"Aggregated response: confidence={aggregated.confidence:.2f}, "
                f"consensus={aggregated.consensus_level:.2f}"
            )

            return aggregated

        except Exception as e:
            logger.error(f"Aggregation error: {e}")
            return self._fallback_aggregation(query, agent_states)

    def _get_system_prompt(self) -> str:
        return """You are an expert response aggregator in a multi-agent AI system.

Your role is to synthesize the perspectives, reasoning, and conclusions from multiple AI agents into a single, coherent, high-quality response.

Your tasks:
1. Identify areas of consensus among agents
2. Resolve conflicts or contradictions
3. Combine the best insights from each agent
4. Produce a clear, accurate final answer
5. Assess overall confidence and consensus level

Be objective, thorough, and favor consensus while acknowledging disagreements."""

    def _build_aggregation_prompt(
        self,
        query: str,
        agent_states: List[AgentState],
        messages: List[Message],
        dialogue_history: str,
    ) -> str:
        prompt = f"""# Multi-Agent Collaboration Summary

## Original Query:
{query}

## Agent Reasoning:

"""

        for i, state in enumerate(agent_states):
            prompt += f"### Agent {state.agent_id} Reasoning:\n"
            prompt += f"{state.reasoning_trace[:500]}...\n\n"

        prompt += f"""## Dialogue Summary:
{dialogue_history[:1000]}...

## Task:
Synthesize the above agent perspectives into a final response. Provide:

1. **FINAL ANSWER**: Clear, concise answer to the original query
2. **CONFIDENCE**: Your confidence in this answer (0-100%)
3. **CONSENSUS**: How much agents agreed (0-100%)
4. **REASONING SUMMARY**: Brief summary of key reasoning points
5. **CONFLICTS** (if any): Note any significant disagreements
6. **AGENT CONTRIBUTIONS**: What each agent contributed

Format your response EXACTLY as:

FINAL_ANSWER: [Your synthesized answer here]

CONFIDENCE: [0-100]

CONSENSUS: [0-100]

REASONING_SUMMARY: [Key points from agents]

CONFLICTS: [Any disagreements, or "None"]

AGENT_0_CONTRIBUTION: [What Agent 0 contributed]
AGENT_1_CONTRIBUTION: [What Agent 1 contributed]
[... for each agent ...]
"""

        return prompt

    def _parse_response(
        self,
        response_text: str,
        agent_states: List[AgentState],
    ) -> AgregatedResponse:
        lines = response_text.split('\n')

        # Extract fields
        final_answer = ""
        confidence = 0.8
        consensus = 0.8
        reasoning_summary = ""
        conflicts = []
        agent_contributions = {}

        for line in lines:
            line = line.strip()

            if line.startswith("FINAL_ANSWER:"):
                final_answer = line.replace("FINAL_ANSWER:", "").strip()
            elif line.startswith("CONFIDENCE:"):
                try:
                    conf_str = line.replace("CONFIDENCE:", "").strip().replace("%", "")
                    confidence = float(conf_str) / 100.0
                except:
                    confidence = 0.8
            elif line.startswith("CONSENSUS:"):
                try:
                    cons_str = line.replace("CONSENSUS:", "").strip().replace("%", "")
                    consensus = float(cons_str) / 100.0
                except:
                    consensus = 0.8
            elif line.startswith("REASONING_SUMMARY:"):
                reasoning_summary = line.replace("REASONING_SUMMARY:", "").strip()
            elif line.startswith("CONFLICTS:"):
                conflict_text = line.replace("CONFLICTS:", "").strip()
                if conflict_text.lower() != "none":
                    conflicts = [conflict_text]
            elif line.startswith("AGENT_") and "_CONTRIBUTION:" in line:
                parts = line.split("_CONTRIBUTION:")
                try:
                    agent_id = int(parts[0].replace("AGENT_", ""))
                    contribution = parts[1].strip()
                    agent_contributions[agent_id] = contribution
                except:
                    pass

        if not final_answer:
            final_answer = response_text[:500]  

        return AgregatedResponse(
            final_answer=final_answer,
            confidence=confidence,
            consensus_level=consensus,
            reasoning_summary=reasoning_summary,
            conflicts=conflicts,
            agent_contributions=agent_contributions,
            metadata={
                "model": self.model,
                "num_agents": len(agent_states),
            }
        )

    def _fallback_aggregation(
        self,
        query: str,
        agent_states: List[AgentState],
    ) -> AgregatedResponse:
        logger.warning("Using fallback aggregation")

        if agent_states:
            first_agent = agent_states[0]
            final_answer = first_agent.reasoning_trace[:300]
        else:
            final_answer = "Unable to generate response due to aggregation error."

        return AgregatedResponse(
            final_answer=final_answer,
            confidence=0.5,
            consensus_level=0.5,
            reasoning_summary="Fallback aggregation used",
            conflicts=["Aggregation error occurred"],
            agent_contributions={i: "Error" for i, _ in enumerate(agent_states)},
            metadata={"fallback": True}
        )

    def aggregate_simple(
        self,
        agent_states: List[AgentState],
    ) -> str:
        if not agent_states:
            return "No agent responses available."

        responses = [state.reasoning_trace[:200] for state in agent_states]
        combined = "\n\n".join([
            f"Agent {state.agent_id}: {resp}..."
            for state, resp in zip(agent_states, responses)
        ])

        return f"Combined agent responses:\n\n{combined}"

    async def close(self):
        if hasattr(self.llm_client, 'close'):
            await self.llm_client.close()
