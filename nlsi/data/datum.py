from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class StandingInstruction:
    nl_instruction: str
    instruction_id: str
    instruction_src: Optional[str] = None


@dataclass(frozen=True)
class NLSIDatum:
    example_id: str
    user_utterance: str
    all_standing_instructions: List[StandingInstruction]
    applicable_standing_instructions: List[StandingInstruction]
    api_calls: List[str]
    metadata: Optional[Dict[str, Any]] = None
    pred_applicable_standing_instructions: Optional[
        List[StandingInstruction]
    ] = None  # predictions
    pred_api_calls: Optional[List[str]] = None
    
    @classmethod
    def from_json(cls, json_data: Dict[str, Any]) -> "NLSIDatum":
        all_standing_instructions = [
            StandingInstruction(**standing_instruction)
            for standing_instruction in json_data["all_standing_instructions"]
        ]
        applicable_standing_instructions = [
            StandingInstruction(**standing_instruction)
            for standing_instruction in json_data["applicable_standing_instructions"]
        ]
        pred_applicable_standing_instructions = json_data.get(
            "pred_applicable_standing_instructions", None
        )
        if pred_applicable_standing_instructions is not None:
            if len(pred_applicable_standing_instructions) > 0:
                if isinstance(pred_applicable_standing_instructions[0], dict):
                    pred_applicable_standing_instructions = [
                        StandingInstruction(**standing_instruction)
                        for standing_instruction in pred_applicable_standing_instructions
                    ]
                else:  # this is to take care of the cases where data is dumped with list of strings.
                    pred_applicable_standing_instructions = [
                        StandingInstruction(standing_instruction, instruction_id=f"{i}")
                        for i, standing_instruction in enumerate(
                            pred_applicable_standing_instructions
                        )
                    ]
        return cls(
            example_id=json_data.get("example_id", None),
            user_utterance=json_data["user_utterance"],
       
            all_standing_instructions=all_standing_instructions,
            applicable_standing_instructions=applicable_standing_instructions,
            api_calls=json_data.get(
                "api_calls", None
            ),  # preserving backward compatibility
            metadata=json_data.get("metadata", None),
            pred_applicable_standing_instructions=pred_applicable_standing_instructions,
            pred_api_calls=json_data.get("pred_api_calls", None),
        )

    @property
    def list_of_applicable_standing_instructions(self) -> List[str]:
        return [
            standing_instruction.nl_instruction
            for standing_instruction in self.applicable_standing_instructions
        ]

    @property
    def list_of_pred_standing_instructions(self) -> List[str]:
        return [
            standing_instruction.nl_instruction
            for standing_instruction in self.pred_applicable_standing_instructions
        ]

  
    @property
    def user_profile(self) -> List[str]:
        return [
            standing_instruction.nl_instruction
            for standing_instruction in self.all_standing_instructions
        ]
