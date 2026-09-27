import os
from jinja2 import Environment, FileSystemLoader
from scenario_lab.schema.scenario_schema import Scenario

ACTOR_DEFAULTS = {
    "cow": (1.5, 2.2), "buffalo": (1.5, 2.2), "dog": (0.4, 0.8),
    "person": (0.5, 0.5), "child": (0.5, 0.5),
    "motorcycle": (0.8, 2.0), "scooter": (0.8, 2.0), "bicycle": (0.6, 1.8),
    "auto_rickshaw": (1.4, 2.8), "e_rickshaw": (1.4, 2.8),
    "pushcart": (1.4, 2.2), "thela": (1.4, 2.2),
    "car": (1.8, 4.2), "suv": (1.8, 4.2),
    "bus": (2.5, 10.0), "truck": (2.5, 8.0), "tractor": (2.2, 5.0)
}

class MatlabAdapter:
    def __init__(self, matlab_dir: str, generated_dir: str):
        self.matlab_dir = matlab_dir
        self.generated_dir = generated_dir
        os.makedirs(self.generated_dir, exist_ok=True)
        self.jinja = Environment(loader=FileSystemLoader(
            os.path.join(os.path.dirname(__file__), 'templates')
        ))

    def generate_script(self, scenario: Scenario) -> tuple[str, str]:
        """
        Returns (script_path, result_json_path).
        Writes generated .m file to generated_dir.
        """
        # Ensure actors have default sizes if not set
        for actor in scenario.actors:
            if actor.width_m is None or actor.length_m is None:
                def_w, def_l = ACTOR_DEFAULTS.get(actor.actor_class, (1.0, 1.0))
                actor.width_m = actor.width_m or def_w
                actor.length_m = actor.length_m or def_l

        func_id = scenario.scenario_id.replace('-', '')[:12]
        result_path = os.path.join(self.generated_dir, f"result_{func_id}.json")
        
        template = self.jinja.get_template('scenario_template.m.jinja2')
        content = template.render(
            s=scenario,
            func_id=func_id,
            output_json_path=result_path.replace('\\', '/')
        )
        
        script_path = os.path.join(self.generated_dir, f"scenario_gen_{func_id}.m")
        with open(script_path, 'w') as f:
            f.write(content)
        
        return script_path, result_path
