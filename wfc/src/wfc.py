import os
import torch
import random
import trimesh

from collections import deque
from typing import Dict, Tuple, Optional


class WaveFunctionCollapseInterface:
    XP = "+x"
    XN = "-x"
    YP = "+y"
    YN = "-y"
    ZP = "+z"
    ZN = "-z"

    directions = [XP, XN, YP, YN, ZP, ZN]
    opposite = {XP: XN, XN: XP, YP: YN, YN: YP, ZP: ZN, ZN: ZP}
    step = {
        XP: (1, 0, 0),
        XN: (-1, 0, 0),
        YP: (0, 1, 0),
        YN: (0, -1, 0),
        ZP: (0, 0, 1),
        ZN: (0, 0, -1),
    }

    FREE = "free"
    OPEN = "open"
    CLOSED = "closed"
    SUPPORT = "support"
    EMPTY = "empty"

    TILE_SIZE = 1

    # you can check the tile modeling in tilegpt\assets\tiles.png
    INTERFACE = {
        0: {
            XP: FREE,
            XN: FREE,
            YP: FREE,
            YN: FREE,
            ZP: EMPTY,
            ZN: EMPTY,
        },
        1: {
            XP: OPEN,
            XN: CLOSED,
            YP: OPEN,
            YN: OPEN,
            ZP: SUPPORT,
            ZN: SUPPORT,
        },
        2: {
            XP: OPEN,
            XN: OPEN,
            YP: CLOSED,
            YN: OPEN,
            ZP: SUPPORT,
            ZN: SUPPORT,
        },
        3: {
            XP: CLOSED,
            XN: OPEN,
            YP: OPEN,
            YN: OPEN,
            ZP: SUPPORT,
            ZN: SUPPORT,
        },
        4: {
            XP: OPEN,
            XN: OPEN,
            YP: OPEN,
            YN: CLOSED,
            ZP: SUPPORT,
            ZN: SUPPORT,
        },
        5: {
            XP: CLOSED,
            XN: CLOSED,
            YP: OPEN,
            YN: OPEN,
            ZP: OPEN,
            ZN: SUPPORT,
        },
        6: {
            XP: OPEN,
            XN: OPEN,
            YP: CLOSED,
            YN: CLOSED,
            ZP: OPEN,
            ZN: SUPPORT,
        },
        7: {
            XP: OPEN,
            XN: OPEN,
            YP: CLOSED,
            YN: CLOSED,
            ZP: SUPPORT,
            ZN: OPEN,
        },
        8: {
            XP: CLOSED,
            XN: CLOSED,
            YP: OPEN,
            YN: OPEN,
            ZP: SUPPORT,
            ZN: OPEN,
        },
        9: {
            XP: OPEN,
            XN: OPEN,
            YP: OPEN,
            YN: OPEN,
            ZP: EMPTY,
            ZN: SUPPORT,
        },
        10: {
            XP: OPEN,
            XN: OPEN,
            YP: CLOSED,
            YN: CLOSED,
            ZP: OPEN,
            ZN: OPEN,
        },
        11: {
            XP: CLOSED,
            XN: CLOSED,
            YP: OPEN,
            YN: OPEN,
            ZP: OPEN,
            ZN: OPEN,
        },
        12: {
            XP: OPEN,
            XN: OPEN,
            YP: OPEN,
            YN: OPEN,
            ZP: SUPPORT,
            ZN: SUPPORT,
        },
        13: {
            XP: OPEN,
            XN: OPEN,
            YP: OPEN,
            YN: OPEN,
            ZP: SUPPORT,
            ZN: SUPPORT,
        },
        14: {
            XP: OPEN,
            XN: OPEN,
            YP: CLOSED,
            YN: CLOSED,
            ZP: OPEN,
            ZN: OPEN,
        },
        15: {
            XP: CLOSED,
            XN: CLOSED,
            YP: OPEN,
            YN: OPEN,
            ZP: OPEN,
            ZN: OPEN,
        },
        16: {
            XP: OPEN,
            XN: SUPPORT,
            YP: CLOSED,
            YN: CLOSED,
            ZP: EMPTY,
            ZN: SUPPORT,
        },
        17: {
            XP: CLOSED,
            XN: CLOSED,
            YP: OPEN,
            YN: SUPPORT,
            ZP: EMPTY,
            ZN: SUPPORT,
        },
        18: {
            XP: SUPPORT,
            XN: OPEN,
            YP: CLOSED,
            YN: CLOSED,
            ZP: EMPTY,
            ZN: SUPPORT,
        },
        19: {
            XP: CLOSED,
            XN: CLOSED,
            YP: CLOSED,
            YN: OPEN,
            ZP: EMPTY,
            ZN: SUPPORT,
        },
    }


class WaveFunctionCollapse(WaveFunctionCollapseInterface):
    def __init__(
        self,
        width: int,
        depth: int,
        height: int,
        seed: int,
        empty_weight: float,
    ):
        self.width = width
        self.depth = depth
        self.height = height
        self.seed = seed
        self.empty_weight = empty_weight

        random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.tiles = sorted(self.INTERFACE.keys())
        self.tile_to_index = {tile: i for i, tile in enumerate(self.tiles)}
        self.index_to_tile = {i: tile for i, tile in enumerate(self.tiles)}

        # self.states
        self._reset()

        self.compatibility = self._compute_compatibility()

        self.cell_weights = torch.ones((1, len(self.tiles)))
        self.cell_weights *= torch.cat(
            [
                torch.tensor([self.empty_weight]),
                torch.ones_like(self.cell_weights[0, 1:]),
            ]
        )
        self.cell_weights = self.cell_weights.squeeze(0)

    def _reset(
        self,
        width: Optional[int] = None,
        depth: Optional[int] = None,
        height: Optional[int] = None,
        seed: Optional[int] = None,
        empty_weight: Optional[float] = None,
    ) -> None:
        """Reset the states of the possible tiles

        Args:
            width (Optional[int], optional): width to re-set. Defaults to None.
            depth (Optional[int], optional): depth to re-set. Defaults to None.
            height (Optional[int], optional): height to re-set. Defaults to None.
            seed (Optional[int], optional): seed to re-set. Defaults to None.
            empty_weight (Optional[float], optional): empty_weight to re-set. Defaults to None.
        """

        if width is not None:
            self.width = width
        if depth is not None:
            self.depth = depth
        if height is not None:
            self.height = height
        if seed is not None:
            self.seed = seed
            random.seed(self.seed)
            torch.manual_seed(self.seed)
        if empty_weight is not None:
            self.empty_weight = empty_weight

        self.states = torch.ones(
            (self.height, self.depth, self.width, len(self.tiles)), dtype=torch.bool
        )

    def _is_compatible(
        self, tile_a_label: str, tile_b_label: str, direction: str
    ) -> bool:
        """Check if the two tiles are compatible in the given direction

        Labels:
            FREE = "free"
            OPEN = "open"
            CLOSED = "closed"
            SUPPORT = "support"
            EMPTY = "empty"

        Args:
            tile_a_label (str): label of the first tile
            tile_b_label (str): label of the second tile
            direction (str): direction to check compatibility

        Returns:
            bool: True if the two tiles are compatible in the given direction otherwise False
        """

        assert tile_a_label in (
            self.FREE,
            self.OPEN,
            self.CLOSED,
            self.SUPPORT,
            self.EMPTY,
        )
        assert tile_b_label in (
            self.FREE,
            self.OPEN,
            self.CLOSED,
            self.SUPPORT,
            self.EMPTY,
        )

        # 'free' label is always compatible when the direction is x or y.
        # in the z-direction, two tiles are compatible only if they have the same label.
        if direction in (self.XP, self.XN, self.YP, self.YN):
            if tile_a_label == self.FREE or tile_b_label == self.FREE:
                return True

        return tile_a_label == tile_b_label

    def _compute_compatibility(self) -> Dict[str, torch.Tensor]:
        """Compute compatibility matrix for each direction

        Example:
            For all tiles, check compatibility between "+x" <-> "-x", "+y" <-> "-y", "+z" <-> "-z"
            The compatibility_matrix is a (t x t) adjacency matrix where `t` is the number of tiles

            >>> compatibility_matrix["+x"].int()
            >>> tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0],
                        [1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0],
                        [1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0],
                        [1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0],
                        [1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0],
                        [1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0],
                        [1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0],
                        [1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0],
                        [1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0],
                        [1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0],
                        [1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1]], dtype=torch.int32)

            >>> torch.all(compatibility["+x"] == compatibility["-x"].t())
            >>> True
            >>> torch.all(compatibility["+y"] == compatibility["-y"].t())
            >>> True
            >>> torch.all(compatibility["+z"] == compatibility["-z"].t())
            >>> True
            >>> compatibility.keys()
            >>> dict_keys(['+x', '-x', '+y', '-y', '+z', '-z'])

        Returns:
            Dict[str, torch.Tensor]: compatibility matrix for each direction
        """

        compatibility = {}
        for direction in self.directions:
            compatibility_matrix = torch.zeros(
                (len(self.tiles), len(self.tiles)), dtype=torch.bool
            )
            direction_opposite = self.opposite[direction]

            for tile_a in self.tiles:
                tile_a_index = self.tile_to_index[tile_a]
                tile_a_label = self.INTERFACE[tile_a][direction]

                for tile_b in self.tiles:
                    tile_b_index = self.tile_to_index[tile_b]
                    tile_b_label = self.INTERFACE[tile_b][direction_opposite]

                    compatibility_matrix[tile_a_index, tile_b_index] = (
                        self._is_compatible(tile_a_label, tile_b_label, direction)
                    )

            compatibility[direction] = compatibility_matrix

        return compatibility

    def _get_possible_neighbors(
        self, possible_tiles: torch.Tensor, direction: str
    ) -> torch.Tensor:
        """Compute possible neighbors of the given tiles with the compatibility matrix

        Args:
            possible_tiles (torch.Tensor): possible tiles
            direction (str): direction of the neighbors

        Returns:
            torch.Tensor: possible neighbors of the given tiles
        """

        possible_neighbors = torch.zeros(possible_tiles.shape[0], dtype=torch.bool)

        for pti in torch.where(possible_tiles == True)[0]:
            possible_neighbors |= self.compatibility[direction][pti]

        return possible_neighbors

    def _is_solved(self) -> bool:
        """Check if the all states of possible cells are collapsed

        Returns:
            bool: True if all states are collapsed, False otherwise
        """

        return torch.all(self.states.sum(dim=-1) == 1).item()

    def _get_min_k_cell(self) -> Tuple[int, int, int] | Tuple[bool, None]:
        """Get the cell with the minimum number of possible tiles

        Returns:
            Tuple[int, int, int]: coordinate of the cell with the minimum possible tiles
            Tuple[bool, None]: False if there is no cell to collapse, None if all cells are collapsed
        """

        min_k = torch.inf
        min_k_cell = None

        for z in range(self.height):
            for y in range(self.depth):
                for x in range(self.width):
                    possible_tiles_count_of_current_cell = (
                        self.states[z, y, x, :].sum().item()
                    )
                    if possible_tiles_count_of_current_cell == 0:
                        return False

                    if possible_tiles_count_of_current_cell == 1:
                        continue

                    if possible_tiles_count_of_current_cell < min_k:
                        min_k = possible_tiles_count_of_current_cell
                        min_k_cell = (x, y, z)

        return min_k_cell

    def _propagate(self, x: int, y: int, z: int) -> bool:
        """Update the possible neighbor tiles of the collapsed cell

        Args:
            x (int): x coordinate of the collapsed cell
            y (int): y coordinate of the collapsed cell
            z (int): z coordinate of the collapsed cell

        Returns:
            bool: True if the cell update succeeds, False otherwise
        """

        queue = deque()
        queue.append((x, y, z))

        while queue:
            x, y, z = queue.popleft()
            possible_tiles = self.states[z, y, x, :]

            if not possible_tiles.any():
                return False

            # traverse all neighbors in the order [+x, -x, +y, -y, +z, -z]
            for direction in self.directions:
                dx, dy, dz = self.step[direction]

                # neighbor coordinates
                ax = x + dx
                ay = y + dy
                az = z + dz

                # check if the coordinates are out of bounds
                if (
                    ax < 0
                    or ax >= self.width
                    or ay < 0
                    or ay >= self.depth
                    or az < 0
                    or az >= self.height
                ):
                    continue

                # compute the possible tiles of the neighbor cell
                possible_neighbors = self._get_possible_neighbors(
                    possible_tiles, direction
                )

                before = self.states[az, ay, ax, :]
                after = before & possible_neighbors

                if not after.any():
                    return False

                if torch.all(before == after):
                    continue

                # update the possible tiles of the neighbor cell
                self.states[az, ay, ax, :] = after
                queue.append((ax, ay, az))

        return True

    def _collapse(self, x: int, y: int, z: int) -> bool:
        """Collapse the cell to a single tile

        Args:
            x (int): x coordinate of the cell
            y (int): y coordinate of the cell
            z (int): z coordinate of the cell

        Returns:
            bool: True if the cell update succeeds, False otherwise
        """

        # self.states[z, y, x, :] means that possible tiles of the cell (height, depth, width, len(tiles))
        # the `empty_weight` of 1 indicates the uniform distribution to collapse
        states_int = self.states[z, y, x, :].float()
        states_int *= self.cell_weights

        tile_index_to_collapse = torch.multinomial(
            states_int / states_int.sum(), 1
        ).item()

        # Set the possible tiles to false, and the selected tile to true
        self.states[z, y, x, :] = False
        self.states[z, y, x, tile_index_to_collapse] = True

        # update possible neighbors of current cell
        propagation_status = self._propagate(x=x, y=y, z=z)

        return propagation_status

    def run(
        self, start_x: int, start_y: int, start_z: int, max_iters: int = 10000, output_directory: Optional[str] = None
    ) -> None:
        """Run the WFC algorithm

        Args:
            start_x (int): x coordinate of the starting cell
            start_y (int): y coordinate of the starting cell
            start_z (int): z coordinate of the starting cell
            max_iters (int, optional): maximum number of iterations. Defaults to 10000.
        """

        cell = (start_x, start_y, start_z)

        for i in range(max_iters):
            if self._is_solved():
                break

            x, y, z = cell

            if not self._collapse(x=x, y=y, z=z):
                # restarts with the initial state if the cell is not collapsed
                self._reset()
                cell = (start_x, start_y, start_z)
                continue

            # get the cell with minimum possible tiles to collapse
            cell = self._get_min_k_cell()

            if cell is False:
                # restarts with the initial state if the cell is not collapsed
                self._reset()
                cell = (start_x, start_y, start_z)
                continue

            if cell is None:
                break
            
            if output_directory is not None:
                self.build(path=os.path.join(output_directory, f"{i}.obj"))
            

    def build(self, path: str) -> None:
        """Build a 3D model based on the wfc states

        Args:
            path (str): path to save the model
        """

        tiles_path = os.path.abspath(os.path.join(__file__, "../../tiles"))

        model = trimesh.Scene()
        for z in range(self.height):
            for y in range(self.depth):
                for x in range(self.width):
                    tile_index = self.states[z, y, x, :].int().argmax().item()
                    tile = self.index_to_tile[tile_index]

                    # empty tile
                    if tile == 0:
                        continue

                    obj_path = os.path.join(tiles_path, f"{tile}.obj")
                    obj = trimesh.load(obj_path)

                    # trimesh uses xzy coordinates system
                    if isinstance(obj, trimesh.Scene):
                        for _, geom in obj.geometry.items():
                            geom.vertices += (
                                x * self.TILE_SIZE,
                                z * self.TILE_SIZE,
                                y * self.TILE_SIZE,
                            )
                            model.add_geometry(geom)

                    else:
                        obj.vertices += (
                            x * self.TILE_SIZE,
                            z * self.TILE_SIZE,
                            y * self.TILE_SIZE,
                        )
                        model.add_geometry(obj)

        if len(model.geometry) > 0:
            model.export(path)
