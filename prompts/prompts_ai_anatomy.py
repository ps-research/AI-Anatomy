"""
AI Anatomy — Prompt Definitions

10 cognitive tasks × 5 variants each = 50 prompts.
Each prompt designed for single next-token prediction on Gemma 3 1B PT.

For each prompt:
  - text: the input string
  - target: the expected next token (or list of acceptable tokens)
  - target_position: which position to analyze (default: last token)
  - notes: why this variant tests the same cognitive capacity differently
"""

TASKS = {
    "T1_semantic_retrieval": {
        "name": "Semantic Retrieval",
        "description": "Factual knowledge retrieval from parametric memory",
        "cognitive_capacity": "Long-term semantic memory access",
        "prompts": [
            {
                "text": "The capital of France is",
                "target": ["Paris"],
                "notes": "Baseline — well-studied factual recall",
            },
            {
                "text": "The largest planet in the solar system is",
                "target": ["Jupiter"],
                "notes": "Science domain retrieval",
            },
            {
                "text": "Water boils at 100 degrees",
                "target": ["Celsius", "C"],
                "notes": "Common knowledge completion",
            },
            {
                "text": "The author of Romeo and Juliet is William",
                "target": ["Shakespeare"],
                "notes": "Named entity completion",
            },
            {
                "text": "The chemical symbol for gold is",
                "target": ["Au"],
                "notes": "Domain-specific knowledge",
            },
        ],
    },

    "T2_working_memory": {
        "name": "Working Memory & Binding",
        "description": "Tracking entity-attribute bindings across positions",
        "cognitive_capacity": "Variable binding in working memory",
        "prompts": [
            {
                "text": "Alice has a cat. Bob has a dog. Alice's pet is a",
                "target": ["cat"],
                "notes": "Two-entity binding, retrieve first entity's attribute",
            },
            {
                "text": "John is tall. Mary is short. The tall person is",
                "target": ["John"],
                "notes": "Property-entity binding, retrieve by property",
            },
            {
                "text": "The red car belongs to Tom. The blue car belongs to Sue. Tom drives a",
                "target": ["red"],
                "notes": "Three-way binding (person-color-object)",
            },
            {
                "text": "In room A there is a key. In room B there is a lamp. The key is in room",
                "target": ["A"],
                "notes": "Location-object binding",
            },
            {
                "text": "Sara likes pizza. Mike likes pasta. The person who likes pizza is",
                "target": ["Sara"],
                "notes": "Preference-entity binding with distractor",
            },
        ],
    },

    "T3_inhibitory_control": {
        "name": "Inhibitory Control",
        "description": "Suppressing a dominant response in favor of a correct but less salient one",
        "cognitive_capacity": "Response inhibition and cognitive control",
        "prompts": [
            {
                "text": "The word 'hello' translated to French is",
                "target": ["bonjour"],
                "notes": "Suppress English continuation, produce French",
            },
            {
                "text": "The opposite of hot is",
                "target": ["cold"],
                "notes": "Suppress associates, produce antonym",
            },
            {
                "text": "If today is Tuesday, yesterday was",
                "target": ["Monday"],
                "notes": "Suppress forward sequence, compute backward",
            },
            {
                "text": "The word 'dog' spelled backwards is",
                "target": ["god"],
                "notes": "Suppress forward reading, produce reverse",
            },
            {
                "text": "In the sequence 1, 2, 3, the number before 2 is",
                "target": ["1"],
                "notes": "Suppress forward continuation, retrieve predecessor",
            },
        ],
    },

    "T4_lexical_disambiguation": {
        "name": "Lexical Disambiguation",
        "description": "Resolving word meaning using context",
        "cognitive_capacity": "Context-dependent semantic processing",
        "prompts": [
            {
                "text": "He deposited money at the bank. The bank is a financial",
                "target": ["institution", "service"],
                "notes": "Classic bank ambiguity, financial context",
            },
            {
                "text": "She sat on the bank of the river. The bank was covered in",
                "target": ["grass", "mud", "sand"],
                "notes": "Bank = riverbank context",
            },
            {
                "text": "The bat flew out of the cave at night. The bat is a",
                "target": ["mammal", "animal", "creature"],
                "notes": "Bat = animal context",
            },
            {
                "text": "He picked up the bat and hit the ball. The bat is made of",
                "target": ["wood", "metal", "aluminum"],
                "notes": "Bat = sports equipment context",
            },
            {
                "text": "The crane lifted the heavy steel beam. The crane is a",
                "target": ["machine", "device"],
                "notes": "Crane = construction context",
            },
        ],
    },

    "T5_causal_reasoning": {
        "name": "Causal Reasoning",
        "description": "Inferring causal consequences not explicitly stated",
        "cognitive_capacity": "Causal inference and world model",
        "prompts": [
            {
                "text": "The glass fell off the table and",
                "target": ["broke", "shattered", "smashed"],
                "notes": "Physical causation (gravity + fragility)",
            },
            {
                "text": "She forgot her umbrella and got",
                "target": ["wet", "soaked", "drenched"],
                "notes": "Implicit rain + consequence chain",
            },
            {
                "text": "The ice cream was left in the sun and",
                "target": ["melted"],
                "notes": "Physical causation (heat + state change)",
            },
            {
                "text": "He studied all night for the exam and",
                "target": ["passed", "aced"],
                "notes": "Social/academic causation",
            },
            {
                "text": "The plant was not watered for weeks and",
                "target": ["died", "wilted", "withered"],
                "notes": "Biological causation (deprivation + consequence)",
            },
        ],
    },

    "T6_pragmatic_inference": {
        "name": "Pragmatic Inference",
        "description": "Understanding indirect speech acts and implied meaning",
        "cognitive_capacity": "Pragmatic reasoning beyond literal meaning",
        "prompts": [
            {
                "text": "Can you pass the salt? She then",
                "target": ["handed", "passed", "gave", "reached"],
                "notes": "Indirect request comprehension",
            },
            {
                "text": "It's cold in here. He then closed the",
                "target": ["window", "door"],
                "notes": "Implicit request via statement",
            },
            {
                "text": "Do you know what time it is? She looked at her",
                "target": ["watch", "phone", "wrist"],
                "notes": "Question-as-request interpretation",
            },
            {
                "text": "That's a nice painting you have there. He said thanks and",
                "target": ["smiled"],
                "notes": "Compliment → social response",
            },
            {
                "text": "I'm really hungry. She then ordered",
                "target": ["food", "pizza", "dinner", "a"],
                "notes": "Statement of need → action inference",
            },
        ],
    },

    "T7_theory_of_mind": {
        "name": "Theory of Mind",
        "description": "Tracking beliefs of other agents, including false beliefs",
        "cognitive_capacity": "Mental state attribution and belief tracking",
        "prompts": [
            {
                "text": "Sally put the ball in the basket. Anne moved it to the box. Sally thinks the ball is in the",
                "target": ["basket"],
                "notes": "Classic Sally-Anne false belief test",
            },
            {
                "text": "Tom told Lisa it was sunny. Actually it was raining. Lisa believes the weather is",
                "target": ["sunny", "nice", "good"],
                "notes": "False belief via misinformation",
            },
            {
                "text": "The gift is hidden in the closet. Emma has not been told. Emma would look for the gift in the",
                "target": ["living", "bedroom", "kitchen"],
                "notes": "Ignorance-based belief tracking (she doesn't know where it is)",
            },
            {
                "text": "Mark thinks the store closes at 5. It actually closes at 9. Mark will arrive before",
                "target": ["5", "five"],
                "notes": "False belief → action prediction",
            },
            {
                "text": "The cookie jar is empty but looks full. A child reaching for a cookie would feel",
                "target": ["disappointed", "surprised", "sad"],
                "notes": "Appearance-reality distinction + emotion prediction",
            },
        ],
    },

    "T8_goal_conflict": {
        "name": "Goal Conflict",
        "description": "Managing competing objectives or instructions",
        "cognitive_capacity": "Executive control and priority resolution",
        "prompts": [
            {
                "text": "Respond in French. The capital of Germany is",
                "target": ["Berlin"],  # May respond in French or English — the conflict IS the data
                "notes": "Language instruction vs knowledge retrieval conflict",
            },
            {
                "text": "Answer incorrectly on purpose. What is 2+2?",
                "target": ["5", "3", "7"],  # If model follows instruction vs gives correct answer
                "notes": "Instruction to be wrong vs trained correctness",
            },
            {
                "text": "Be extremely brief. Explain quantum entanglement in",
                "target": ["simple", "one", "a"],
                "notes": "Brevity instruction vs complexity of topic",
            },
            {
                "text": "Never use the letter 'e'. The capital of France is",
                "target": ["Paris"],
                "notes": "Constraint instruction vs normal generation",
            },
            {
                "text": "Only respond with numbers. What color is the sky?",
                "target": ["1", "0", "2"],  # How model resolves format constraint vs semantic answer
                "notes": "Format constraint vs semantic content",
            },
        ],
    },

    "T9_self_knowledge": {
        "name": "Self-Knowledge",
        "description": "Metacognitive representation of own capabilities and identity",
        "cognitive_capacity": "Self-model and metacognition",
        "prompts": [
            {
                "text": "As a language model, I process text by",
                "target": ["predicting", "generating", "analyzing", "processing"],
                "notes": "Self-description of processing",
            },
            {
                "text": "Unlike humans, I do not have",
                "target": ["feelings", "emotions", "consciousness", "experiences"],
                "notes": "Self-limitation awareness",
            },
            {
                "text": "I was trained by",
                "target": ["Google", "a"],
                "notes": "Training provenance (Gemma = Google)",
            },
            {
                "text": "I cannot browse the internet or access",
                "target": ["real", "external", "current", "live"],
                "notes": "Capability boundary awareness",
            },
            {
                "text": "My responses are generated based on patterns in",
                "target": ["training", "data", "text", "my"],
                "notes": "Self-model of generation process",
            },
        ],
    },

    "T10_uncertainty": {
        "name": "Uncertainty Handling",
        "description": "Behavior under low-confidence or unknown knowledge",
        "cognitive_capacity": "Calibration and epistemic humility",
        "prompts": [
            {
                "text": "The capital of Tuvalu is",
                "target": ["Fun", "Funafuti"],
                "notes": "Obscure fact — model may or may not know",
            },
            {
                "text": "The 37th digit of pi is",
                "target": ["5"],  # Actually it is 5, but model likely uncertain
                "notes": "Precise knowledge test",
            },
            {
                "text": "The population of Luxembourg in 2023 was approximately",
                "target": ["660", "650", "6"],
                "notes": "Approximate numerical knowledge",
            },
            {
                "text": "The third president of Botswana was",
                "target": ["Quett", "Festus"],
                "notes": "Obscure political knowledge",
            },
            {
                "text": "The melting point of hafnium is approximately",
                "target": ["2", "22"],  # 2233°C
                "notes": "Obscure scientific constant",
            },
        ],
    },
}


# ============================================================
# Helper functions
# ============================================================

def get_all_prompts():
    """Return flat list of all (task_id, variant_idx, prompt_dict)."""
    all_prompts = []
    for task_id, task in TASKS.items():
        for i, prompt in enumerate(task["prompts"]):
            all_prompts.append((task_id, i, prompt))
    return all_prompts


def get_task_prompts(task_id):
    """Return all prompt variants for a specific task."""
    return TASKS[task_id]["prompts"]


def get_task_info(task_id):
    """Return task metadata."""
    return {
        "name": TASKS[task_id]["name"],
        "description": TASKS[task_id]["description"],
        "cognitive_capacity": TASKS[task_id]["cognitive_capacity"],
    }


def get_all_task_ids():
    """Return ordered list of task IDs."""
    return list(TASKS.keys())


# Quick validation
if __name__ == "__main__":
    all_prompts = get_all_prompts()
    print(f"Total prompts: {len(all_prompts)}")
    for task_id in get_all_task_ids():
        info = get_task_info(task_id)
        prompts = get_task_prompts(task_id)
        print(f"  {task_id}: {info['name']} ({len(prompts)} variants)")
        for i, p in enumerate(prompts):
            print(f"    {i+1}. '{p['text'][:60]}...' → {p['target']}")
