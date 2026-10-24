"""Run this ONCE per environment (scripts/common_ee.py documents why): it sets
up Earth Engine authentication and initialisation. Opens a browser; sign in with
the Google account tied to the project registered for this work (one project
only - decision D5)."""
if __name__ == "__main__":
    from common_ee import ROOT, ee  # noqa: F401

    ee.Authenticate()
    from no2map.config import load_config

    project = load_config()["quota"]["project_id"]
    if project is None:
        print("Now edit configs/default.yaml -> quota.project_id and re-run ee_init via any probe.")
    else:
        ee.Initialize(project=project)
        print("Earth Engine initialised for project:", project)
