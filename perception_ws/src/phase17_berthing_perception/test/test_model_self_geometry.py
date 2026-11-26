"""验证实测发动机回波与船旁外部结构不会被同一个大矩形混淆。"""

from pathlib import Path
import numpy as np

from phase16z_s3_runtime.s3_ownship_geometry import load_ownship_geometry
from phase16z_s3_runtime.s3_candidate_repair import load_footprint_yaml
from phase17_berthing_perception.environment_core import extract_cloud_evidence, load_footprint_polygons
from phase17_berthing_perception.grid_core import GridConfig


CONFIG = Path(__file__).resolve().parents[1] / "config"


def test_observed_rear_echoes_belong_to_engines():
    model = load_ownship_geometry(CONFIG / "visual_self_geometry.yaml")
    assert model.contains([[-2.56238, 1.01078, .50247], [-2.56678, -1.01317, .49881]]).all()


def test_external_surface_above_float_is_not_ownship():
    model = load_ownship_geometry(CONFIG / "visual_self_geometry.yaml")
    assert not model.contains([[2.25, 1.03, 1.1], [2., 1.4, 1.2], [-2.8, 1.5, 1.2]]).any()


def test_observed_noise_tail_on_deck_is_ownship():
    model = load_ownship_geometry(CONFIG / "visual_self_geometry.yaml")
    assert model.contains([[-.847671, .258207, 1.357957], [-.00874, -.793083, .78845],
                           [-.771736, -.020067, 1.51079]]).all()


def test_full_collision_model_covers_engine_and_bow():
    footprint = load_footprint_yaml(CONFIG / "hull_collision_footprint_base_link.yaml")
    assert footprint.contains(-2.7, 1.03)
    assert footprint.contains(2.4, 1.03)
    assert not footprint.contains(2., 1.5)


def test_nearby_external_wall_survives_full_model_filter():
    yy, zz = np.meshgrid(np.linspace(1.35, 1.65, 30), np.linspace(.9, 1.7, 30))
    points = np.column_stack((np.full(yy.size, 2.), yy.ravel(), zz.ravel()))
    occupied, _, _, _ = extract_cloud_evidence(points, width=30, height=30,
        transform=np.eye(4), footprint_polygons=load_footprint_polygons(CONFIG / "hull_collision_footprint_base_link.yaml"),
        self_geometry=load_ownship_geometry(CONFIG / "visual_self_geometry.yaml"), cfg=GridConfig())
    assert len(occupied) > 500
