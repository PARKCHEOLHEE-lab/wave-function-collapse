import os
import sys
import tqdm
import random

from typing import Tuple, Optional

from wfc import WaveFunctionCollapse


def get_random_parameters(
    width_min: int = 3,
    width_max: int = 4,
    depth_min: int = 2,
    depth_max: int = 4,
    height_min: int = 3,
    height_max: int = 6,
    empty_weight_min: float = 0.7,
    empty_weight_max: float = 1.5,
) -> Tuple[int, int, int, int, int, int, float, float]:
    """Get random parameters for the wave function collapse

    Args:
        width_min (int, optional): width min bound. Defaults to 2.
        width_max (int, optional): width max bound. Defaults to 4.
        depth_min (int, optional): depth min bound. Defaults to 2.
        depth_max (int, optional): depth max bound. Defaults to 4.
        height_min (int, optional): height min bound. Defaults to 2.
        height_max (int, optional): height max bound. Defaults to 10.
        empty_weight_min (float, optional): empty weight min bound. Defaults to 0.5.
        empty_weight_max (float, optional): empty weight max bound. Defaults to 1.5.

    Returns:
        Tuple[int, int, int, int, int, int, float, float]: random parameters
    """

    width = random.randint(width_min, width_max)
    depth = random.randint(depth_min, depth_max)
    height = random.randint(height_min, height_max)
    seed = random.randint(0, sys.maxsize)

    t = random.random()
    empty_weight = (1 - t) * empty_weight_min + t * empty_weight_max

    return {
        "width": width,
        "depth": depth,
        "height": height,
        "seed": seed,
        "empty_weight": empty_weight,
    }


def main(
    num_generation: int = 1,
    intermediate_build: bool = False,
    width: Optional[int] = None,
    depth: Optional[int] = None,
    height: Optional[int] = None,
    seed: Optional[int] = None,
    empty_weight: Optional[float] = None,
):
    output_directory = os.path.abspath(os.path.join(__file__, "../../output"))
    os.makedirs(output_directory, exist_ok=True)

    wfc_parameters = get_random_parameters()
    if width is not None:
        wfc_parameters["width"] = width
    if depth is not None:
        wfc_parameters["depth"] = depth
    if height is not None:
        wfc_parameters["height"] = height
    if empty_weight is not None:
        wfc_parameters["empty_weight"] = empty_weight
    if seed is not None:
        wfc_parameters["seed"] = seed

    wfc = WaveFunctionCollapse(**wfc_parameters)

    for g in tqdm.tqdm(range(num_generation)):
        result_directory = os.path.join(output_directory, str(g))
        os.makedirs(result_directory, exist_ok=True)

        wfc.run(
            start_x=random.randint(0, wfc.width - 1),
            start_y=random.randint(0, wfc.depth - 1),
            start_z=0,
            output_directory=result_directory if intermediate_build else None,
        )

        wfc.build(path=os.path.join(result_directory, "result.obj"))

        wfc_parameters = get_random_parameters()
        if width is not None:
            wfc_parameters["width"] = width
        if depth is not None:
            wfc_parameters["depth"] = depth
        if height is not None:
            wfc_parameters["height"] = height
        if empty_weight is not None:
            wfc_parameters["empty_weight"] = empty_weight
        if seed is not None:
            wfc_parameters["seed"] = seed

        wfc._reset(**wfc_parameters)


if __name__ == "__main__":
    main(
        num_generation=3,
        intermediate_build=False,
        width=None,
        depth=None,
        height=None,
        seed=None,
        empty_weight=None,
    )
