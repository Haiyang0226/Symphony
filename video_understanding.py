from typing import Dict, Any, List, Optional
import json
import time
import logging
from agents.core_agent import CoreAgent
from agents.A_PerceptionAgent import PerceptionAgent
from agents.A_SubtitleAgent import SubtitleAgent
from agents.A_LocalizeAgent import LocalizeAgent
from agents.A_ReflectionAgent import ReflectionAgent

from utils import fix_and_parse_json

class VideoUnderstandingSystem:
    def __init__(
        self,
        video_duration: float,
        question: str,
        frame_path: str,
        sub_path: str,
        log_path: str,
        data_name: str,
        max_cycles: int = 17,
    ):
        self.video_duration = video_duration
        self.question = question
        self.frame_path = frame_path
        self.sub_path = sub_path
        self.data_name = data_name
        self.logger = self.get_logger(log_path)
        self.if_reflected = 0
        
        self.max_cycles = max_cycles
        
        self.cycle_count = 0
        self.completed = False
        self.final_answer = None
        
        self.history = []
        
        self.core_agent = CoreAgent(
            question=self.question,
            data_name=self.data_name,
            video_duration=self.video_duration,
            logger=self.logger,
        )
        
        self.perception_agent = PerceptionAgent(
            frame_path=self.frame_path,
            data_name=self.data_name,
            logger=self.logger,
        )
        
        self.subtitle_agent = SubtitleAgent(
            question=self.question,
            subtitle_path=self.sub_path,
            data_name=self.data_name,
            logger=self.logger,
        )
        
        self.reflection_agent = ReflectionAgent(
            question=self.question,
            data_name=self.data_name,
            logger=self.logger,
        )

        self.localize_agent = LocalizeAgent(
            video_duration=self.video_duration,
            frame_path=self.frame_path,
            question=self.question, # Pass the question
            data_name=self.data_name,
            logger=self.logger,
        )

    def get_logger(self, log_path):
        logger = logging.getLogger(f"{__name__}.{id(self)}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.FileHandler(log_path, encoding='utf-8')
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger

    def run(self) -> Dict:
        while not self.completed and self.cycle_count < self.max_cycles:
            self.cycle_count += 1
            self.logger.info(f"\n=== Cycle {self.cycle_count} ===")

            # 1. Core Agent Decides
            self.logger.info("--- Core Agent Deciding ---")
            core_decision_str = self.core_agent.run(history=self.history)
            try:
                # core_decision = json.loads(core_decision_str)
                core_decision = fix_and_parse_json(core_decision_str, self.logger)
            except json.JSONDecodeError:
                self.logger.error(f"CoreAgent output is not a valid JSON: {core_decision_str}")
                # Handle error, maybe retry or fail
                return self._build_final_result("failed", "CoreAgent returned invalid JSON.")

            self.logger.info(f"Core Agent decision: {core_decision}")

            agent_to_call = core_decision.get("agent")
            result = None

            # 2. Execute Chosen Agent
            if agent_to_call == "PerceptionAgent":
                self.logger.info("--- Calling Perception Agent ---")
                instruction = core_decision.get("instruct")
                result = self.perception_agent.run(instruct=instruction, question=self.question, video_duration=self.video_duration)
                
            elif agent_to_call == "SubtitleAgent":
                self.logger.info("--- Calling Subtitle Agent ---")
                # instruction = core_decision.get("instruct")
                result = self.subtitle_agent.run()

            elif agent_to_call == "LocalizeAgent":
                self.logger.info("--- Calling Localize Agent ---")
                # instruction = core_decision.get("instruct")
                result = self.localize_agent.run()

            elif agent_to_call == "finish":
                self.logger.info("--- Core Agent proposed a final answer. Starting Reflection ---")
                proposed_answer = core_decision.get("answer")

                self.history.append({"action": agent_to_call, "reason": core_decision.get("reason"), "answer": core_decision.get("answer")})
                
                # Call ReflectionAgent for assessment
                if not self.if_reflected:
                    assessment = self.reflection_agent.run(proposed_answer=proposed_answer, history = self.history)
                    self.if_reflected = 1
                else:
                    assessment = {
                        "credible": True,
                    }
                
                if assessment.get("credible"):
                    self.logger.info("--- Reflection Agent found the answer credible. Finishing Task. ---")
                    self.completed = True
                    self.final_answer = proposed_answer
                    self.history.append({
                        "action": "reflection", 
                        "assessment": "credible", 
                        "proposed_answer": proposed_answer
                    })
                    self.history.append({"action": "finish", "answer": self.final_answer})
                    break
                else:
                    self.logger.warning("--- Reflection Agent found issues. Continuing cycle. ---")
                    comment = assessment.get("comment", "No comment provided.")
                    self.history.append({
                        "action": "Upon reflection, your current answer is not reliable. Please reconsider carefully and reorganize the relevant information to provide a credible response.", 
                        "assessment": "not_credible", 
                        "comment": comment, 
                    })
                    # The loop will continue, and CoreAgent will get the new history
                    continue
            
            else:
                self.logger.warning(f"Unknown agent requested: {agent_to_call}")
                # Handle unknown agent, maybe add to history and continue
                self.history.append({"action": "unknown_agent", "decision": core_decision})
                continue

            # 3. Record History
            if agent_to_call != "finish":
                if core_decision.get("instruct") == None:
                    self.history.append({"action": agent_to_call, "reason": core_decision.get("reason"), "result": result})
                else:
                    self.history.append({"action": agent_to_call, "instruct": core_decision.get("instruct"), "reason": core_decision.get("reason"), "result": result})
            self.logger.info(f"Result: {result}")

        if self.completed:
            return self._build_final_result("completed", "Task finished successfully.")
        else:
            return self._build_final_result("failed", "Exceeded maximum cycles.")

    def _build_final_result(self, status: str, reason: str) -> Dict:
        return {
            "status": status,
            "answer": self.final_answer,
            "reason": reason,
            "cycles": self.cycle_count,
            "history": self.history,
        }

if __name__ == "__main__":
    Vus = VideoUnderstandingSystem(
        video_duration=3669,
        question="What year appears in the opening caption of the video?\n(A) 1636\n(B) 1366\n(C) 1363\n(D) 1633",
        frame_path="/path/to/your/frames",
        sub_path="/path/to/your/subs.json",
        log_path="./test.log",
        data_name="example_video"
    )
    final_result = Vus.run()
    print(json.dumps(final_result, indent=4))