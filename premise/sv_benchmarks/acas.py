# %%
import math
from typing import Callable, Optional

import numpy as np
from PIL import Image, ImageDraw
from stormvogel import pgc, ModelType, Model

from premise.sv_benchmarks.coarse import CoarseValue, CoarseGaussianValue

NMAC = 150  # m (Near Mid Air Collision)

RADIUS_MIN = NMAC  # meters
RADIUS_MAX = 2000  # old 15 000  # meters
RADIUS_COARSE = 30  # 71
RADIUS_OBS = 10

BEARING_MIN = -math.pi  # radians
BEARING_MAX = math.pi  # radians
BEARING_COARSE = 40  # old 121
BEARING_OBS = 10
BEARING_STD = np.deg2rad(10)  # old np.deg2rad(1.0027)  # radians

REL_HEADING_MIN = -math.pi  # radians
REL_HEADING_MAX = math.pi  # radians
REL_HEADING_COARSE = 10  # old 121
REL_HEADING_OBS = 5
REL_HEADING_STD = np.deg2rad(10)  # old np.deg2rad(1.0027)  # radians

EGO_SPEED_MIN = 15  # m/s
EGO_SPEED_MAX = 75  # m/s
EGO_SPEED_COARSE = 4  # old 94
EGO_SPEED_OBS = 3
EGO_SPEED_STD = 5  # old 0.5  # m/s

INT_SPEED_MIN = 0  # m/s
INT_SPEED_MAX = 75  # m/s
INT_SPEED_COARSE = 4  # old 4
INT_SPEED_OBS = 3
INT_SPEED_STD = 10  # old 1.1  # m/s

TIMESTEP = 2  # s

MIN_PROB_VAL = 0.01


class ACAState(pgc.State):
    radius: CoarseValue
    bearing: CoarseGaussianValue
    rel_heading: CoarseGaussianValue
    ego_speed: CoarseGaussianValue
    int_speed: CoarseGaussianValue
    nmac: float
    timestep: int
    init: bool

    def __init__(
        self,
        radius: Optional[CoarseValue] = None,
        bearing: Optional[CoarseGaussianValue] = None,
        rel_heading: Optional[CoarseGaussianValue] = None,
        ego_speed: Optional[CoarseGaussianValue] = None,
        int_speed: Optional[CoarseGaussianValue] = None,
        radius_min=RADIUS_MIN,
        radius_max=RADIUS_MAX,
        radius_coarse=RADIUS_COARSE,
        radius_obs=RADIUS_OBS,
        bearing_min=BEARING_MIN,
        bearing_max=BEARING_MAX,
        bearing_coarse=BEARING_COARSE,
        bearing_obs=BEARING_OBS,
        bearing_std=BEARING_STD,
        rel_heading_min=REL_HEADING_MIN,
        rel_heading_max=REL_HEADING_MAX,
        rel_heading_coarse=REL_HEADING_COARSE,
        rel_heading_obs=REL_HEADING_OBS,
        rel_heading_std=REL_HEADING_STD,
        ego_speed_min=EGO_SPEED_MIN,
        ego_speed_max=EGO_SPEED_MAX,
        ego_speed_coarse=EGO_SPEED_COARSE,
        ego_speed_obs=EGO_SPEED_OBS,
        ego_speed_std=EGO_SPEED_STD,
        int_speed_min=INT_SPEED_MIN,
        int_speed_max=INT_SPEED_MAX,
        int_speed_coarse=INT_SPEED_COARSE,
        int_speed_obs=INT_SPEED_OBS,
        int_speed_std=INT_SPEED_STD,
        min_prob_val=MIN_PROB_VAL,
        timestep=TIMESTEP,
        nmac=NMAC,
    ):
        self.nmac = nmac
        self.timestep = timestep

        if (
            radius is None
            and bearing is None
            and rel_heading is None
            and ego_speed is None
            and int_speed is None
        ):
            self.init = True
        else:
            self.init = False

        if radius is None:
            self.radius = CoarseValue(
                radius_min, radius_max, radius_coarse, radius_obs, timestep
            )
        else:
            self.radius = radius

        if bearing is None:
            self.bearing = CoarseGaussianValue(
                bearing_min,
                bearing_max,
                bearing_coarse,
                bearing_obs,
                timestep,
                0,
                bearing_std,
                min_prob_val,
                loop=True,
            )
        else:
            self.bearing = bearing

        if rel_heading is None:
            self.rel_heading = CoarseGaussianValue(
                rel_heading_min,
                rel_heading_max,
                rel_heading_coarse,
                rel_heading_obs,
                timestep,
                0,
                rel_heading_std,
                min_prob_val,
            )
        else:
            self.rel_heading = rel_heading

        if ego_speed is None:
            self.ego_speed = CoarseGaussianValue(
                ego_speed_min,
                ego_speed_max,
                ego_speed_coarse,
                ego_speed_obs,
                timestep,
                0,
                ego_speed_std,
                min_prob_val,
            )
        else:
            self.ego_speed = ego_speed

        if int_speed is None:
            self.int_speed = CoarseGaussianValue(
                int_speed_min,
                int_speed_max,
                int_speed_coarse,
                int_speed_obs,
                timestep,
                0,
                int_speed_std,
                min_prob_val,
            )
        else:
            self.int_speed = int_speed

    def calc(self):
        x = (
            self.radius.value * np.cos(self.bearing.value)
            - self.ego_speed.value * self.timestep
            + self.int_speed.value * self.timestep * np.cos(self.rel_heading.value)
        )
        y = self.radius.value * np.sin(
            self.bearing.value
        ) + self.int_speed.value * self.timestep * np.sin(self.rel_heading.value)
        r_new = np.hypot(x, y)
        bearing_new = np.arctan2(y, x)
        return r_new, bearing_new

    def labels(self):
        labels = []
        if self.init:
            return ["init"]

        if self.radius.value <= self.nmac + 1:
            labels.append("nmac")
        if self.radius.at_max():
            labels.append("out of radius")
        labels.append(str(self.valuations()))
        return labels

    def obs(self):
        if self.init:
            return (True,)
        else:
            return (
                False,
                self.radius.obs,
                self.bearing.obs,
                self.rel_heading.obs,
                self.ego_speed.obs,
                self.int_speed.obs,
            )

    def valuations(self):
        if self.init:
            return {
                "init": True,
                "radius": 0.0,
                "bearing": 0.0,
                "rel_heading": 0.0,
                "ego_speed": 0.0,
                "int_speed": 0.0,
                "ACAState": self,
            }
        else:
            return {
                "init": False,
                "radius": self.radius.value,
                "bearing": self.bearing.value,
                "rel_heading": self.rel_heading.value,
                "ego_speed": self.ego_speed.value,
                "int_speed": self.int_speed.value,
                "ACAState": self,
            }

    def draw(self) -> Image.Image:
        w, h = 2048, 2048
        ego_size = 20
        int_size = 20
        line_width = 4

        scale = (w / 2) / self.radius.max_bound

        img = Image.new("RGBA", (w, h), "black")
        draw = ImageDraw.Draw(img)

        for b in range(self.bearing.coarseness):
            for r in range(self.radius.coarseness, 0, -1):
                start_angle = (
                    self.bearing.calc_value(b - 1) + self.bearing.calc_value(b)
                ) / 2
                end_angle = (
                    self.bearing.calc_value(b) + self.bearing.calc_value(b + 1)
                ) / 2
                c = ((b + r) % 2) * 2 + 2
                radius = (self.radius.calc_value(r) + self.radius.calc_value(r - 1)) / 2
                draw.pieslice(
                    (
                        w // 2 - (radius * scale),
                        h // 2 - (radius * scale),
                        w // 2 + (radius * scale),
                        h // 2 + (radius * scale),
                    ),
                    np.rad2deg(start_angle),
                    np.rad2deg(end_angle),
                    # outline="green",
                    fill=f"#0{c:X}0",
                )

        # Draw outlines of observation slices
        for b in range(self.bearing.obs_coarseness):
            for r in range(self.radius.obs_coarseness + 1, 0, -1):
                start_angle = (
                    self.bearing.calc_value(b - 1, obs=True)
                    + self.bearing.calc_value(b, obs=True)
                ) / 2
                end_angle = (
                    self.bearing.calc_value(b, obs=True)
                    + self.bearing.calc_value(b + 1, obs=True)
                ) / 2
                c = ((b + r) % 2) * 2 + 2
                radius = (
                    self.radius.calc_value(r, obs=True)
                    + self.radius.calc_value(r - 1, obs=True)
                ) / 2
                draw.pieslice(
                    (
                        w // 2 - (radius * scale),
                        h // 2 - (radius * scale),
                        w // 2 + (radius * scale),
                        h // 2 + (radius * scale),
                    ),
                    np.rad2deg(start_angle),
                    np.rad2deg(end_angle),
                    outline=f"#00{c:X}",
                    width=line_width,
                )

        draw.circle((w // 2, h // 2), w // 2, outline=f"#0F0", width=line_width)

        draw.circle(
            (w // 2, h // 2),
            self.nmac * scale,
            width=line_width,
            outline="red",
            fill="black",
        )

        # Draw ego plane
        draw.regular_polygon(
            (w // 2, h // 2, ego_size),
            3,
            rotation=-90,
            fill="#8F8",
        )

        if not self.init:
            speed = self.ego_speed.value * self.timestep * scale
            draw.line(
                (
                    w // 2,
                    h // 2,
                    w // 2 + speed,
                    h // 2,
                ),
                width=line_width,
                fill="orange",
            )

            int_x = self.radius.value * np.cos(self.bearing.value) * scale
            int_y = self.radius.value * np.sin(self.bearing.value) * scale
            draw.regular_polygon(
                (int_x + w // 2, int_y + h // 2, int_size),
                3,
                rotation=-90 - np.rad2deg(self.rel_heading.value),
                fill="red" if self.radius.at_min() else "#8F8",
            )
            int_obs_x = self.radius.obs * np.cos(self.bearing.obs) * scale
            int_obs_y = self.radius.obs * np.sin(self.bearing.obs) * scale
            draw.regular_polygon(
                (int_obs_x + w // 2, int_obs_y + h // 2, int_size),
                3,
                rotation=-90 - np.rad2deg(self.rel_heading.obs),
                fill="#880000" if self.radius.at_min() else "#8888FF",
            )

            int_speed_x = (
                self.int_speed.value
                * np.cos(self.rel_heading.value)
                * scale
                * self.timestep
            )
            int_speed_y = (
                self.int_speed.value
                * np.sin(self.rel_heading.value)
                * scale
                * self.timestep
            )
            draw.line(
                (
                    w // 2 + int_x,
                    h // 2 + int_y,
                    w // 2 + int_x + int_speed_x - speed,
                    h // 2 + int_y + int_speed_y,
                ),
                width=line_width,
                fill="blue",
            )
            draw.line(
                (
                    w // 2 + int_x,
                    h // 2 + int_y,
                    w // 2 + int_x + int_speed_x,
                    h // 2 + int_y + int_speed_y,
                ),
                width=line_width,
                fill="orange",
            )

            draw.multiline_text(
                (0, 0),
                f"r={self.radius.value:.2f}m\nb={np.rad2deg(self.bearing.value):.2f}°\n"
                f"h={np.rad2deg(self.rel_heading.value):.2f}°\n"
                f"v={self.ego_speed.value:.2f}m/s\niv={self.int_speed.value:.2f}m/s",
                fill="white",
                font_size=30,
            )

        return img

    def __hash__(self):
        return hash(
            (
                hash(self.radius),
                hash(self.bearing),
                hash(self.rel_heading),
                hash(self.ego_speed),
                hash(self.int_speed),
            )
        )

    def __eq__(self, other):
        if not isinstance(other, ACAState):
            return False
        return (
            self.radius == other.radius
            and self.bearing == other.bearing
            and self.rel_heading == other.rel_heading
            and self.ego_speed == other.ego_speed
            and self.int_speed == other.int_speed
        )


class ObservationBuilder:
    obs_map: dict[tuple, int] = {}
    obs_func: Callable

    def __init__(self, obs_func: Callable):
        self.obs_func = obs_func

    def __call__(self, state) -> int:
        obs = self.obs_func(state)
        if obs not in self.obs_map:
            self.obs_map[obs] = len(self.obs_map)
        return self.obs_map[obs]


def acas_delta_builder(**kwargs):
    def acas_delta(state: ACAState, action: pgc.Action):
        if state.init:
            distr = []

            rs = state.radius.unif_distr()
            bs = state.bearing.unif_distr()
            rhs = state.rel_heading.unif_distr()
            ess = state.ego_speed.unif_distr()
            iss = state.int_speed.unif_distr()
            print(
                f"taking {len(rs) * len(bs) * len(rhs) * len(ess) * len(iss)} initial states"
            )
            for p_r, r in rs:
                for p_b, bearing in bs:
                    for p_rh, rel_heading in rhs:
                        for p_es, ego_speed in ess:
                            for p_is, int_speed in iss:
                                distr.append(
                                    (
                                        p_r * p_b * p_rh * p_es * p_is,
                                        ACAState(
                                            r,
                                            bearing,
                                            rel_heading,
                                            ego_speed,
                                            int_speed,
                                            **kwargs,
                                        ),
                                    )
                                )
        elif state.radius.at_max():
            return [(1, state)]
        else:
            r, b = state.calc()
            radius = state.radius.copy_update(r)
            if radius.at_min():
                return [(1, state)]

            distr = []
            bs = state.bearing.distr(b)
            rhs = state.rel_heading.distr(state.rel_heading.value)
            egs = state.ego_speed.distr(state.ego_speed.value)
            iss = state.int_speed.distr(state.int_speed.value)
            for p_b, bearing in bs:
                for p_rh, rel_heading in rhs:
                    for p_es, ego_speed in egs:
                        for p_is, int_speed in iss:
                            distr.append(
                                (
                                    p_b * p_rh * p_es * p_is,
                                    ACAState(
                                        radius,
                                        bearing,
                                        rel_heading,
                                        ego_speed,
                                        int_speed,
                                        **kwargs,
                                    ),
                                )
                            )

        return distr

    return acas_delta


def build_acas_model(**kwargs) -> Model:
    init_state = ACAState(**kwargs)
    obs_builder = ObservationBuilder(lambda s: s.obs())
    return pgc.build_pgc(
        acas_delta_builder(**kwargs),
        init_state,
        labels=lambda s: s.labels(),
        valuations=lambda s: s.valuations(),
        available_actions=lambda s: [pgc.Action([])],
        observations=obs_builder,
        max_size=200000,
        modeltype=ModelType.POMDP,
        safe=False,
    )
