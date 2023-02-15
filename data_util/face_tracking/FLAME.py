# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms
# in the LICENSE file included with this software distribution.
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import pickle

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.transforms import Transform3d
from RigNeRF.models.lbs import (batch_rodrigues, lbs, rot_mat_to_euler,
                                vertices2landmarks)


def to_tensor(array, dtype=torch.float32):
    if "torch.tensor" not in str(type(array)):
        return torch.tensor(array, dtype=dtype)


def to_np(array, dtype=np.float32):
    if "scipy.sparse" in str(type(array)):
        array = array.todense()

    return np.array(array, dtype=dtype)


PYT_ROT_Y = torch.from_numpy(
    np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
).float()[None, :]
T_ROT_Y = Transform3d()
T_ROT_Y._matrix = PYT_ROT_Y

ROT_Z = torch.from_numpy(
    np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
).float()[None, :]
TO_PYT3D = Transform3d()
TO_PYT3D._matrix = ROT_Z


class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


class FLAMEAux(nn.Module):
    """
    borrowed from https://github.com/soubhiksanyal/FLAME_PyTorch/blob/master/FLAME.py
    Given flame parameters this class generates a differentiable FLAME function
    which outputs the a mesh and 2D/3D facial landmarks
    """

    def __init__(self, config):
        super(FLAMEAux, self).__init__()
        with open(config.flame_model_path, "rb") as f:
            ss = pickle.load(f, encoding="latin1")
            flame_model = Struct(**ss)

        self.dtype = torch.float32
        self.register_buffer(
            "faces_tensor",
            to_tensor(to_np(flame_model.f, dtype=np.int64), dtype=torch.long),
        )
        # The vertices of the template model
        self.register_buffer(
            "v_template", to_tensor(to_np(flame_model.v_template), dtype=self.dtype)
        )
        # The shape components and expression
        shapedirs = to_tensor(to_np(flame_model.shapedirs), dtype=self.dtype)
        shapedirs = torch.cat(
            [
                shapedirs[:, :, : config.n_shape],
                shapedirs[:, :, 300 : 300 + config.n_exp],
            ],
            2,
        )
        self.register_buffer("shapedirs", shapedirs)
        # The pose components
        num_pose_basis = flame_model.posedirs.shape[-1]
        posedirs = np.reshape(flame_model.posedirs, [-1, num_pose_basis]).T
        self.register_buffer("posedirs", to_tensor(to_np(posedirs), dtype=self.dtype))
        #
        self.register_buffer(
            "J_regressor", to_tensor(to_np(flame_model.J_regressor), dtype=self.dtype)
        )
        parents = to_tensor(to_np(flame_model.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer("parents", parents)
        self.register_buffer(
            "lbs_weights", to_tensor(to_np(flame_model.weights), dtype=self.dtype)
        )

        # Fixing Eyeball and neck rotation
        default_eyball_pose = torch.zeros([1, 6], dtype=self.dtype, requires_grad=False)
        self.eye_pose = default_eyball_pose
        # self.register_parameter(
        #    "eye_pose", nn.Parameter(default_eyball_pose, requires_grad=False)
        # )
        default_neck_pose = torch.zeros([1, 3], dtype=self.dtype, requires_grad=False)
        self.neck_pose = default_neck_pose
        # self.register_parameter(
        #    "neck_pose", nn.Parameter(default_neck_pose, requires_grad=False)
        # )

        # Static and Dynamic Landmark embeddings for FLAME
        lmk_embeddings = np.load(
            config.flame_lmk_embedding_path, allow_pickle=True, encoding="latin1"
        )
        lmk_embeddings = lmk_embeddings[()]
        self.register_buffer(
            "lmk_faces_idx",
            torch.from_numpy(lmk_embeddings["static_lmk_faces_idx"]).long(),
        )
        self.register_buffer(
            "lmk_bary_coords",
            torch.from_numpy(lmk_embeddings["static_lmk_bary_coords"]).to(self.dtype),
        )
        self.register_buffer(
            "dynamic_lmk_faces_idx", lmk_embeddings["dynamic_lmk_faces_idx"].long()
        )
        self.register_buffer(
            "dynamic_lmk_bary_coords",
            lmk_embeddings["dynamic_lmk_bary_coords"].to(self.dtype),
        )
        self.register_buffer(
            "full_lmk_faces_idx",
            torch.from_numpy(lmk_embeddings["full_lmk_faces_idx"]).long(),
        )
        self.register_buffer(
            "full_lmk_bary_coords",
            torch.from_numpy(lmk_embeddings["full_lmk_bary_coords"]).to(self.dtype),
        )

        neck_kin_chain = []
        NECK_IDX = 1
        curr_idx = torch.tensor(NECK_IDX, dtype=torch.long)

        while curr_idx != -1:
            neck_kin_chain.append(curr_idx)
            curr_idx = self.parents[curr_idx]
        self.register_buffer("neck_kin_chain", torch.stack(neck_kin_chain))

    def _find_dynamic_lmk_idx_and_bcoords(
        self,
        pose,
        dynamic_lmk_faces_idx,
        dynamic_lmk_b_coords,
        neck_kin_chain,
        dtype=torch.float32,
    ):
        """
        Selects the face contour depending on the reletive position of the head
        Input:
            vertices: N X num_of_vertices X 3
            pose: N X full pose
            dynamic_lmk_faces_idx: The list of contour face indexes
            dynamic_lmk_b_coords: The list of contour barycentric weights
            neck_kin_chain: The tree to consider for the relative rotation
            dtype: Data type
        return:
            The contour face indexes and the corresponding barycentric weights
        """

        batch_size = pose.shape[0]

        aa_pose = torch.index_select(pose.view(batch_size, -1, 3), 1, neck_kin_chain)
        rot_mats = batch_rodrigues(aa_pose.view(-1, 3), dtype=dtype).view(
            batch_size, -1, 3, 3
        )

        rel_rot_mat = (
            torch.eye(3, device=pose.device, dtype=dtype)
            .unsqueeze_(dim=0)
            .expand(batch_size, -1, -1)
        )

        for idx in range(len(neck_kin_chain)):
            rel_rot_mat = torch.bmm(rot_mats[:, idx], rel_rot_mat)

        y_rot_angle = torch.round(
            torch.clamp(rot_mat_to_euler(rel_rot_mat) * 180.0 / np.pi, max=39)
        ).to(dtype=torch.long)

        neg_mask = y_rot_angle.lt(0).to(dtype=torch.long)
        mask = y_rot_angle.lt(-39).to(dtype=torch.long)
        neg_vals = mask * 78 + (1 - mask) * (39 - y_rot_angle)
        y_rot_angle = neg_mask * neg_vals + (1 - neg_mask) * y_rot_angle

        dyn_lmk_faces_idx = torch.index_select(dynamic_lmk_faces_idx, 0, y_rot_angle)
        dyn_lmk_b_coords = torch.index_select(dynamic_lmk_b_coords, 0, y_rot_angle)

        return dyn_lmk_faces_idx, dyn_lmk_b_coords

    def _vertices2landmarks(self, vertices, faces, lmk_faces_idx, lmk_bary_coords):
        """
        Calculates landmarks by barycentric interpolation
        Input:
            vertices: torch.tensor NxVx3, dtype = torch.float32
                The tensor of input vertices
            faces: torch.tensor (N*F)x3, dtype = torch.long
                The faces of the mesh
            lmk_faces_idx: torch.tensor N X L, dtype = torch.long
                The tensor with the indices of the faces used to calculate the
                landmarks.
            lmk_bary_coords: torch.tensor N X L X 3, dtype = torch.float32
                The tensor of barycentric coordinates that are used to interpolate
                the landmarks

        Returns:
            landmarks: torch.tensor NxLx3, dtype = torch.float32
                The coordinates of the landmarks for each mesh in the batch
        """
        # Extract the indices of the vertices for each face
        # NxLx3
        batch_size, num_verts = vertices.shape[:2]
        lmk_faces = (
            torch.index_select(faces, 0, lmk_faces_idx.view(-1))
            .view(1, -1, 3)
            .view(batch_size, lmk_faces_idx.shape[1], -1)
        )

        lmk_faces += (
            torch.arange(batch_size, dtype=torch.long)
            .view(-1, 1, 1)
            .to(device=vertices.device)
            * num_verts
        )

        lmk_vertices = vertices.view(-1, 3)[lmk_faces]
        landmarks = torch.einsum("blfi,blf->bli", [lmk_vertices, lmk_bary_coords])

        return landmarks

    def seletec_3d68(self, vertices):
        landmarks3d = vertices2landmarks(
            vertices,
            self.faces_tensor,
            self.full_lmk_faces_idx.repeat(vertices.shape[0], 1),
            self.full_lmk_bary_coords.repeat(vertices.shape[0], 1, 1),
        )

        return landmarks3d

    def forward(
        self,
        shape_params=None,
        expression_params=None,
        pose_params=None,
        eye_pose_params=None,
        translation=None,
        global_rot=None,
        per_vertex_def=None,
    ):
        """
        Input:
            shape_params: N X number of shape parameters
            expression_params: N X number of expression parameters
            pose_params: N X number of pose parameters (6)
        return:d
            vertices: N X V X 3
            landmarks: N X number of landmarks X 3
        """
        batch_size = shape_params.shape[0]

        if global_rot is not None:
            global_rot_mat = batch_rodrigues(rot_vecs=global_rot)
        else:
            global_rot_mat = None

        if eye_pose_params is None:
            eye_pose_params = self.eye_pose.expand(batch_size, -1).to(
                shape_params.device
            )
        betas = torch.cat([shape_params, expression_params], dim=1)

        if pose_params.shape[1] == 6:
            full_pose = torch.cat(
                [
                    pose_params[:, :3],  # GLobal Rotation
                    self.neck_pose.expand(batch_size, -1).to(
                        shape_params.device
                    ),  # Neck pose
                    pose_params[:, 3:],  # Jaw pose
                    eye_pose_params,  # Eye pose
                ],
                dim=1,
            )
        elif pose_params.shape[1] == 9:
            full_pose = torch.cat(
                [
                    pose_params,
                    eye_pose_params,  # Eye pose
                ],
                dim=1,
            )
        else:
            raise ValueError
        template_vertices = self.v_template.unsqueeze(0).expand(batch_size, -1, -1)

        if per_vertex_def is not None:
            template_vertices = template_vertices + per_vertex_def

        vertices, _, v_shaped, v_posed, pose_offsets = lbs(
            betas,
            full_pose,
            template_vertices,
            self.shapedirs,
            self.posedirs,
            self.J_regressor,
            self.parents,
            self.lbs_weights,
            dtype=self.dtype,
            return_pose_offsets=True,
        )

        if global_rot_mat is not None:
            assert len(vertices.shape) == 3
            # num_batches = vertices.shape[0]
            # _vertices_list = []

            # for i in range(num_batches):
            #    curr_rot_mat = global_rot_mat[i][None]
            #    curr_rot_mat = einops.repeat(
            #        curr_rot_mat, "b h w -> (repeat b) h w", repeat=vertices.shape[1]
            #    )
            #    curr_vertices = einops.rearrange(vertices[i], "n d -> n 1 d")
            #    _vertices = torch.bmm(curr_vertices, curr_rot_mat)
            #    _vertices_list.append(_vertices[:, 0, :])
            # vertices = torch.stack(_vertices_list, dim=0)

            global_rot_mat = einops.rearrange(global_rot_mat, "b h w -> b w h")
            assert len(vertices.shape) == 3
            assert len(global_rot_mat.shape) == 3
            assert vertices.shape[0] == global_rot_mat.shape[0]
            vertices = torch.matmul(vertices, global_rot_mat)

        if translation is None:
            translation = torch.zeros((1, vertices.shape[1], 3)).to(vertices.device)
        else:
            translation = translation[:, None, ...].repeat(1, vertices.shape[1], 1)

        vertices = vertices + translation

        lmk_faces_idx = self.lmk_faces_idx.unsqueeze(dim=0).expand(batch_size, -1)
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(dim=0).expand(
            batch_size, -1, -1
        )

        dyn_lmk_faces_idx, dyn_lmk_bary_coords = self._find_dynamic_lmk_idx_and_bcoords(
            full_pose,
            self.dynamic_lmk_faces_idx,
            self.dynamic_lmk_bary_coords,
            self.neck_kin_chain,
            dtype=self.dtype,
        )
        lmk_faces_idx = torch.cat([dyn_lmk_faces_idx, lmk_faces_idx], 1)
        lmk_bary_coords = torch.cat([dyn_lmk_bary_coords, lmk_bary_coords], 1)

        landmarks2d = vertices2landmarks(
            vertices, self.faces_tensor, lmk_faces_idx, lmk_bary_coords
        )
        bz = vertices.shape[0]
        landmarks3d = vertices2landmarks(
            vertices,
            self.faces_tensor,
            self.full_lmk_faces_idx.repeat(bz, 1),
            self.full_lmk_bary_coords.repeat(bz, 1, 1),
        )

        return vertices, landmarks2d, landmarks3d, v_shaped, v_posed, pose_offsets


class FLAMELearn(nn.Module):
    """
    borrowed from https://github.com/soubhiksanyal/FLAME_PyTorch/blob/master/FLAME.py
    Given flame parameters this class generates a differentiable FLAME function
    which outputs the a mesh and 2D/3D facial landmarks
    """

    def __init__(self, config, args, device=torch.device("cuda:0")):
        super(FLAMELearn, self).__init__()
        with open(config.flame_model_path, "rb") as f:
            ss = pickle.load(f, encoding="latin1")
            flame_model = Struct(**ss)

        self.dtype = torch.float32
        self.register_buffer(
            "faces_tensor",
            to_tensor(to_np(flame_model.f, dtype=np.int64), dtype=torch.long),
        )
        # The vertices of the template model
        self.register_buffer(
            "v_template", to_tensor(to_np(flame_model.v_template), dtype=self.dtype)
        )
        # The shape components and expression
        shapedirs = to_tensor(to_np(flame_model.shapedirs), dtype=self.dtype)
        shapedirs = torch.cat(
            [
                shapedirs[:, :, : config.n_shape],
                shapedirs[:, :, 300 : 300 + config.n_exp],
            ],
            2,
        )
        self.register_buffer("shapedirs", shapedirs)
        _verts, faces, aux = load_obj(config.topology_path)
        self.faces_verts_idx = faces.verts_idx.to(device)
        self.template_verts_def = nn.Parameter(1e-18 * torch.ones_like(_verts[0:1]))

        if args.shape_opt_basis_dim >= 0 and args.exp_opt_basis_dim >= 0:
            if (
                args.shape_opt_basis_dim < config.n_shape
                and args.shape_opt_basis_dim > 0
            ):
                self._shapedirs_delta_opt = nn.Parameter(
                    1e-16
                    * torch.zeros_like(
                        self.shapedirs[..., : int(args.shape_opt_basis_dim)]
                    )
                )
                self.register_buffer(
                    "shapedirs_delta_const",
                    torch.zeros_like(
                        self.shapedirs[..., int(args.shape_opt_basis_dim) : 300]
                    ),
                )
            elif args.shape_opt_basis_dim == 0:
                self._shapedirs_delta_opt = None
                self.register_buffer(
                    "shapedirs_delta_const",
                    torch.zeros_like(self.shapedirs[..., : config.n_shape]),
                )
            elif int(args.shape_opt_basis_dim) == config.n_shape:
                self._shapedirs_delta_opt = nn.Parameter(
                    torch.zeros_like(self.shapedirs[..., : int(config.n_shape)])
                )
                self.shapedirs_delta_const = None
            else:
                raise ValueError

            if args.exp_opt_basis_dim < config.n_exp and args.exp_opt_basis_dim > 0:
                self._expdirs_delta_opt = nn.Parameter(
                    1e-16
                    * torch.randn_like(
                        self.shapedirs[..., 300 : 300 + int(args.exp_opt_basis_dim)]
                    )
                )
                self.register_buffer(
                    "expdirs_delta_const",
                    torch.zeros_like(
                        self.shapedirs[
                            ...,
                            300 + int(args.exp_opt_basis_dim) : 300 + config.n_exp,
                        ]
                    ),
                )
            elif args.exp_opt_basis_dim == 0:
                self._expdirs_delta_opt = None
                self.register_buffer(
                    "expdirs_delta_const",
                    torch.zeros_like(
                        self.shapedirs[..., 300 : 300 + int(config.n_exp)]
                    ),
                )
            elif int(args.exp_opt_basis_dim) == config.n_exp:
                self._expdirs_delta_opt = nn.Parameter(
                    torch.zeros_like(self.shapedirs[..., 300 : 300 + int(config.n_exp)])
                )
                self.expdirs_delta_const = None
            else:
                raise ValueError
        else:
            raise ValueError
        # The pose components
        num_pose_basis = flame_model.posedirs.shape[-1]

        posedirs = np.reshape(flame_model.posedirs, [-1, num_pose_basis]).T
        self.register_buffer("posedirs", to_tensor(to_np(posedirs), dtype=self.dtype))
        #
        self.posedirs_delta = nn.Parameter(1e-16 * torch.randn_like(self.posedirs))
        self.register_buffer(
            "J_regressor", to_tensor(to_np(flame_model.J_regressor), dtype=self.dtype)
        )
        self.j_regressor_delta = nn.Parameter(
            1e-16 * torch.randn_like(self.J_regressor)
        )
        parents = to_tensor(to_np(flame_model.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer("parents", parents)

        self.register_buffer(
            "lbs_weights", to_tensor(to_np(flame_model.weights), dtype=self.dtype)
        )
        self.lbs_weights_delta = nn.Parameter(
            1e-16 * torch.randn_like(self.lbs_weights)
        )

        # Fixing Eyeball and neck rotation
        default_eyball_pose = torch.zeros([1, 6], dtype=self.dtype, requires_grad=False)
        self.eye_pose = default_eyball_pose
        # self.register_parameter(
        #    "eye_pose", nn.Parameter(default_eyball_pose, requires_grad=False)
        # )
        default_neck_pose = torch.zeros([1, 3], dtype=self.dtype, requires_grad=False)
        self.neck_pose = default_neck_pose
        # self.register_parameter(
        #    "neck_pose", nn.Parameter(default_neck_pose, requires_grad=False)
        # )

        # Static and Dynamic Landmark embeddings for FLAME
        lmk_embeddings = np.load(
            config.flame_lmk_embedding_path, allow_pickle=True, encoding="latin1"
        )
        lmk_embeddings = lmk_embeddings[()]
        self.register_buffer(
            "lmk_faces_idx",
            torch.from_numpy(lmk_embeddings["static_lmk_faces_idx"]).long(),
        )
        self.register_buffer(
            "lmk_bary_coords",
            torch.from_numpy(lmk_embeddings["static_lmk_bary_coords"]).to(self.dtype),
        )
        self.register_buffer(
            "dynamic_lmk_faces_idx", lmk_embeddings["dynamic_lmk_faces_idx"].long()
        )
        self.register_buffer(
            "dynamic_lmk_bary_coords",
            lmk_embeddings["dynamic_lmk_bary_coords"].to(self.dtype),
        )
        self.register_buffer(
            "full_lmk_faces_idx",
            torch.from_numpy(lmk_embeddings["full_lmk_faces_idx"]).long(),
        )
        self.register_buffer(
            "full_lmk_bary_coords",
            torch.from_numpy(lmk_embeddings["full_lmk_bary_coords"]).to(self.dtype),
        )

        neck_kin_chain = []
        NECK_IDX = 1
        curr_idx = torch.tensor(NECK_IDX, dtype=torch.long)

        while curr_idx != -1:
            neck_kin_chain.append(curr_idx)
            curr_idx = self.parents[curr_idx]
        self.register_buffer("neck_kin_chain", torch.stack(neck_kin_chain))

    def _find_dynamic_lmk_idx_and_bcoords(
        self,
        pose,
        dynamic_lmk_faces_idx,
        dynamic_lmk_b_coords,
        neck_kin_chain,
        dtype=torch.float32,
    ):
        """
        Selects the face contour depending on the reletive position of the head
        Input:
            vertices: N X num_of_vertices X 3
            pose: N X full pose
            dynamic_lmk_faces_idx: The list of contour face indexes
            dynamic_lmk_b_coords: The list of contour barycentric weights
            neck_kin_chain: The tree to consider for the relative rotation
            dtype: Data type
        return:
            The contour face indexes and the corresponding barycentric weights
        """

        batch_size = pose.shape[0]

        aa_pose = torch.index_select(pose.view(batch_size, -1, 3), 1, neck_kin_chain)
        rot_mats = batch_rodrigues(aa_pose.view(-1, 3), dtype=dtype).view(
            batch_size, -1, 3, 3
        )

        rel_rot_mat = (
            torch.eye(3, device=pose.device, dtype=dtype)
            .unsqueeze_(dim=0)
            .expand(batch_size, -1, -1)
        )

        for idx in range(len(neck_kin_chain)):
            rel_rot_mat = torch.bmm(rot_mats[:, idx], rel_rot_mat)

        y_rot_angle = torch.round(
            torch.clamp(rot_mat_to_euler(rel_rot_mat) * 180.0 / np.pi, max=39)
        ).to(dtype=torch.long)

        neg_mask = y_rot_angle.lt(0).to(dtype=torch.long)
        mask = y_rot_angle.lt(-39).to(dtype=torch.long)
        neg_vals = mask * 78 + (1 - mask) * (39 - y_rot_angle)
        y_rot_angle = neg_mask * neg_vals + (1 - neg_mask) * y_rot_angle

        dyn_lmk_faces_idx = torch.index_select(dynamic_lmk_faces_idx, 0, y_rot_angle)
        dyn_lmk_b_coords = torch.index_select(dynamic_lmk_b_coords, 0, y_rot_angle)

        return dyn_lmk_faces_idx, dyn_lmk_b_coords

    def _vertices2landmarks(self, vertices, faces, lmk_faces_idx, lmk_bary_coords):
        """
        Calculates landmarks by barycentric interpolation
        Input:
            vertices: torch.tensor NxVx3, dtype = torch.float32
                The tensor of input vertices
            faces: torch.tensor (N*F)x3, dtype = torch.long
                The faces of the mesh
            lmk_faces_idx: torch.tensor N X L, dtype = torch.long
                The tensor with the indices of the faces used to calculate the
                landmarks.
            lmk_bary_coords: torch.tensor N X L X 3, dtype = torch.float32
                The tensor of barycentric coordinates that are used to interpolate
                the landmarks

        Returns:
            landmarks: torch.tensor NxLx3, dtype = torch.float32
                The coordinates of the landmarks for each mesh in the batch
        """
        # Extract the indices of the vertices for each face
        # NxLx3
        batch_size, num_verts = vertices.shape[:2]
        lmk_faces = (
            torch.index_select(faces, 0, lmk_faces_idx.view(-1))
            .view(1, -1, 3)
            .view(batch_size, lmk_faces_idx.shape[1], -1)
        )

        lmk_faces += (
            torch.arange(batch_size, dtype=torch.long)
            .view(-1, 1, 1)
            .to(device=vertices.device)
            * num_verts
        )

        lmk_vertices = vertices.view(-1, 3)[lmk_faces]
        landmarks = torch.einsum("blfi,blf->bli", [lmk_vertices, lmk_bary_coords])

        return landmarks

    def seletec_3d68(self, vertices):
        landmarks3d = vertices2landmarks(
            vertices,
            self.faces_tensor,
            self.full_lmk_faces_idx.repeat(vertices.shape[0], 1),
            self.full_lmk_bary_coords.repeat(vertices.shape[0], 1, 1),
        )

        return landmarks3d

    def forward(
        self,
        shape_params=None,
        expression_params=None,
        pose_params=None,
        eye_pose_params=None,
        translation=None,
        global_rot=None,
        per_vertex_def=None,
    ):
        """
        Input:
            shape_params: N X number of shape parameters
            expression_params: N X number of expression parameters
            pose_params: N X number of pose parameters (6)
        return:d
            vertices: N X V X 3
            landmarks: N X number of landmarks X 3
        """
        batch_size = shape_params.shape[0]

        if global_rot is not None:
            global_rot_mat = batch_rodrigues(rot_vecs=global_rot)
        else:
            global_rot_mat = None

        if eye_pose_params is None:
            eye_pose_params = self.eye_pose.expand(batch_size, -1).to(
                shape_params.device
            )
        betas = torch.cat([shape_params, expression_params], dim=1)

        if self.shapedirs_delta_const is None:
            shapedirs_delta_opt = self._shapedirs_delta_opt
        elif self._shapedirs_delta_opt is None:
            shapedirs_delta_opt = self.shapedirs_delta_const
        else:
            shapedirs_delta_opt = torch.cat(
                [
                    self._shapedirs_delta_opt,
                    self.shapedirs_delta_const,
                ],
                dim=-1,
            )

        if self.expdirs_delta_const is None:
            expdirs_delta_opt = self._expdirs_delta_opt
        elif self._expdirs_delta_opt is None:
            expdirs_delta_opt = self.expdirs_delta_const
        else:
            expdirs_delta_opt = torch.cat(
                [
                    self._expdirs_delta_opt,
                    self.expdirs_delta_const,
                ],
                dim=-1,
            )
        self.shapedirs_delta = torch.cat(
            [shapedirs_delta_opt, expdirs_delta_opt], dim=-1
        )

        if pose_params.shape[1] == 6:
            full_pose = torch.cat(
                [
                    pose_params[:, :3],  # GLobal Rotation
                    self.neck_pose.expand(batch_size, -1).to(
                        shape_params.device
                    ),  # Neck pose
                    pose_params[:, 3:],  # Jaw pose
                    eye_pose_params,  # Eye pose
                ],
                dim=1,
            )
        elif pose_params.shape[1] == 9:
            full_pose = torch.cat(
                [
                    pose_params,
                    eye_pose_params,  # Eye pose
                ],
                dim=1,
            )
        else:
            raise ValueError
        _template_vertices = self.v_template.unsqueeze(0).expand(batch_size, -1, -1)
        verts_def_batch = self.template_verts_def.repeat(batch_size, 1, 1)
        faces_verts_idx = self.faces_verts_idx.repeat(batch_size, 1, 1)
        _meshes = Meshes(verts=_template_vertices, faces=faces_verts_idx)
        normals = _meshes.verts_normals_padded()
        template_vertices = _template_vertices + verts_def_batch * normals

        if per_vertex_def is not None:
            template_vertices = template_vertices + per_vertex_def
        vertices, *_ = lbs(
            betas,
            full_pose,
            template_vertices,
            self.shapedirs + self.shapedirs_delta,
            self.posedirs + self.posedirs_delta,
            self.J_regressor + self.j_regressor_delta,
            self.parents,
            self.lbs_weights + self.lbs_weights_delta,
            dtype=self.dtype,
        )

        if global_rot_mat is not None:
            assert len(vertices.shape) == 3
            # num_batches = vertices.shape[0]
            # _vertices_list = []

            # for i in range(num_batches):
            #    curr_rot_mat = global_rot_mat[i][None]
            #    curr_rot_mat = einops.repeat(
            #        curr_rot_mat, "b h w -> (repeat b) h w", repeat=vertices.shape[1]
            #    )
            #    curr_vertices = einops.rearrange(vertices[i], "n d -> n 1 d")
            #    _vertices = torch.bmm(curr_vertices, curr_rot_mat)
            #    _vertices_list.append(_vertices[:, 0, :])
            # vertices = torch.stack(_vertices_list, dim=0)

            global_rot_mat = einops.rearrange(global_rot_mat, "b h w -> b w h")
            assert len(vertices.shape) == 3
            assert len(global_rot_mat.shape) == 3
            assert vertices.shape[0] == global_rot_mat.shape[0]
            vertices = torch.matmul(vertices, global_rot_mat)

        if translation is None:
            translation = torch.zeros((1, vertices.shape[1], 3)).to(vertices.device)
        else:
            translation = translation[:, None, ...].repeat(1, vertices.shape[1], 1)

        vertices = vertices + translation

        lmk_faces_idx = self.lmk_faces_idx.unsqueeze(dim=0).expand(batch_size, -1)
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(dim=0).expand(
            batch_size, -1, -1
        )

        dyn_lmk_faces_idx, dyn_lmk_bary_coords = self._find_dynamic_lmk_idx_and_bcoords(
            full_pose,
            self.dynamic_lmk_faces_idx,
            self.dynamic_lmk_bary_coords,
            self.neck_kin_chain,
            dtype=self.dtype,
        )
        lmk_faces_idx = torch.cat([dyn_lmk_faces_idx, lmk_faces_idx], 1)
        lmk_bary_coords = torch.cat([dyn_lmk_bary_coords, lmk_bary_coords], 1)

        landmarks2d = vertices2landmarks(
            vertices, self.faces_tensor, lmk_faces_idx, lmk_bary_coords
        )
        bz = vertices.shape[0]
        landmarks3d = vertices2landmarks(
            vertices,
            self.faces_tensor,
            self.full_lmk_faces_idx.repeat(bz, 1),
            self.full_lmk_bary_coords.repeat(bz, 1, 1),
        )

        return vertices, landmarks2d, landmarks3d


class FLAME(nn.Module):
    """
    borrowed from https://github.com/soubhiksanyal/FLAME_PyTorch/blob/master/FLAME.py
    Given flame parameters this class generates a differentiable FLAME function
    which outputs the a mesh and 2D/3D facial landmarks
    """

    def __init__(self, config):
        super(FLAME, self).__init__()
        with open(config.flame_model_path, "rb") as f:
            ss = pickle.load(f, encoding="latin1")
            flame_model = Struct(**ss)

        self.dtype = torch.float32
        self.register_buffer(
            "faces_tensor",
            to_tensor(to_np(flame_model.f, dtype=np.int64), dtype=torch.long),
        )
        # The vertices of the template model
        self.register_buffer(
            "v_template", to_tensor(to_np(flame_model.v_template), dtype=self.dtype)
        )
        # The shape components and expression
        shapedirs = to_tensor(to_np(flame_model.shapedirs), dtype=self.dtype)
        shapedirs = torch.cat(
            [
                shapedirs[:, :, : config.n_shape],
                shapedirs[:, :, 300 : 300 + config.n_exp],
            ],
            2,
        )
        self.register_buffer("shapedirs", shapedirs)
        # The pose components
        num_pose_basis = flame_model.posedirs.shape[-1]
        posedirs = np.reshape(flame_model.posedirs, [-1, num_pose_basis]).T
        self.register_buffer("posedirs", to_tensor(to_np(posedirs), dtype=self.dtype))
        #
        self.register_buffer(
            "J_regressor", to_tensor(to_np(flame_model.J_regressor), dtype=self.dtype)
        )
        parents = to_tensor(to_np(flame_model.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer("parents", parents)
        self.register_buffer(
            "lbs_weights", to_tensor(to_np(flame_model.weights), dtype=self.dtype)
        )

        # Fixing Eyeball and neck rotation
        default_eyball_pose = torch.zeros([1, 6], dtype=self.dtype, requires_grad=False)
        self.eye_pose = default_eyball_pose
        # self.register_parameter(
        #    "eye_pose", nn.Parameter(default_eyball_pose, requires_grad=False)
        # )
        default_neck_pose = torch.zeros([1, 3], dtype=self.dtype, requires_grad=False)
        self.neck_pose = default_neck_pose
        # self.register_parameter(
        #    "neck_pose", nn.Parameter(default_neck_pose, requires_grad=False)
        # )

        # Static and Dynamic Landmark embeddings for FLAME
        lmk_embeddings = np.load(
            config.flame_lmk_embedding_path, allow_pickle=True, encoding="latin1"
        )
        lmk_embeddings = lmk_embeddings[()]
        self.register_buffer(
            "lmk_faces_idx",
            torch.from_numpy(lmk_embeddings["static_lmk_faces_idx"]).long(),
        )
        self.register_buffer(
            "lmk_bary_coords",
            torch.from_numpy(lmk_embeddings["static_lmk_bary_coords"]).to(self.dtype),
        )
        self.register_buffer(
            "dynamic_lmk_faces_idx", lmk_embeddings["dynamic_lmk_faces_idx"].long()
        )
        self.register_buffer(
            "dynamic_lmk_bary_coords",
            lmk_embeddings["dynamic_lmk_bary_coords"].to(self.dtype),
        )
        self.register_buffer(
            "full_lmk_faces_idx",
            torch.from_numpy(lmk_embeddings["full_lmk_faces_idx"]).long(),
        )
        self.register_buffer(
            "full_lmk_bary_coords",
            torch.from_numpy(lmk_embeddings["full_lmk_bary_coords"]).to(self.dtype),
        )

        neck_kin_chain = []
        NECK_IDX = 1
        curr_idx = torch.tensor(NECK_IDX, dtype=torch.long)

        while curr_idx != -1:
            neck_kin_chain.append(curr_idx)
            curr_idx = self.parents[curr_idx]
        self.register_buffer("neck_kin_chain", torch.stack(neck_kin_chain))

    def _find_dynamic_lmk_idx_and_bcoords(
        self,
        pose,
        dynamic_lmk_faces_idx,
        dynamic_lmk_b_coords,
        neck_kin_chain,
        dtype=torch.float32,
    ):
        """
        Selects the face contour depending on the reletive position of the head
        Input:
            vertices: N X num_of_vertices X 3
            pose: N X full pose
            dynamic_lmk_faces_idx: The list of contour face indexes
            dynamic_lmk_b_coords: The list of contour barycentric weights
            neck_kin_chain: The tree to consider for the relative rotation
            dtype: Data type
        return:
            The contour face indexes and the corresponding barycentric weights
        """

        batch_size = pose.shape[0]

        aa_pose = torch.index_select(pose.view(batch_size, -1, 3), 1, neck_kin_chain)
        rot_mats = batch_rodrigues(aa_pose.view(-1, 3), dtype=dtype).view(
            batch_size, -1, 3, 3
        )

        rel_rot_mat = (
            torch.eye(3, device=pose.device, dtype=dtype)
            .unsqueeze_(dim=0)
            .expand(batch_size, -1, -1)
        )

        for idx in range(len(neck_kin_chain)):
            rel_rot_mat = torch.bmm(rot_mats[:, idx], rel_rot_mat)

        y_rot_angle = torch.round(
            torch.clamp(rot_mat_to_euler(rel_rot_mat) * 180.0 / np.pi, max=39)
        ).to(dtype=torch.long)

        neg_mask = y_rot_angle.lt(0).to(dtype=torch.long)
        mask = y_rot_angle.lt(-39).to(dtype=torch.long)
        neg_vals = mask * 78 + (1 - mask) * (39 - y_rot_angle)
        y_rot_angle = neg_mask * neg_vals + (1 - neg_mask) * y_rot_angle

        dyn_lmk_faces_idx = torch.index_select(dynamic_lmk_faces_idx, 0, y_rot_angle)
        dyn_lmk_b_coords = torch.index_select(dynamic_lmk_b_coords, 0, y_rot_angle)

        return dyn_lmk_faces_idx, dyn_lmk_b_coords

    def _vertices2landmarks(self, vertices, faces, lmk_faces_idx, lmk_bary_coords):
        """
        Calculates landmarks by barycentric interpolation
        Input:
            vertices: torch.tensor NxVx3, dtype = torch.float32
                The tensor of input vertices
            faces: torch.tensor (N*F)x3, dtype = torch.long
                The faces of the mesh
            lmk_faces_idx: torch.tensor N X L, dtype = torch.long
                The tensor with the indices of the faces used to calculate the
                landmarks.
            lmk_bary_coords: torch.tensor N X L X 3, dtype = torch.float32
                The tensor of barycentric coordinates that are used to interpolate
                the landmarks

        Returns:
            landmarks: torch.tensor NxLx3, dtype = torch.float32
                The coordinates of the landmarks for each mesh in the batch
        """
        # Extract the indices of the vertices for each face
        # NxLx3
        batch_size, num_verts = vertices.shape[:2]
        lmk_faces = (
            torch.index_select(faces, 0, lmk_faces_idx.view(-1))
            .view(1, -1, 3)
            .view(batch_size, lmk_faces_idx.shape[1], -1)
        )

        lmk_faces += (
            torch.arange(batch_size, dtype=torch.long)
            .view(-1, 1, 1)
            .to(device=vertices.device)
            * num_verts
        )

        lmk_vertices = vertices.view(-1, 3)[lmk_faces]
        landmarks = torch.einsum("blfi,blf->bli", [lmk_vertices, lmk_bary_coords])

        return landmarks

    def seletec_3d68(self, vertices):
        landmarks3d = vertices2landmarks(
            vertices,
            self.faces_tensor,
            self.full_lmk_faces_idx.repeat(vertices.shape[0], 1),
            self.full_lmk_bary_coords.repeat(vertices.shape[0], 1, 1),
        )

        return landmarks3d

    def get_verts_landmarks(self, vertices):
        bz = vertices.shape[0]
        landmarks3d = vertices2landmarks(
            vertices,
            self.faces_tensor,
            self.full_lmk_faces_idx.repeat(bz, 1),
            self.full_lmk_bary_coords.repeat(bz, 1, 1),
        )

        return landmarks3d

    def forward(
        self,
        shape_params=None,
        expression_params=None,
        pose_params=None,
        eye_pose_params=None,
        translation=None,
        global_rot=None,
        per_vertex_def=None,
    ):
        """
        Input:
            shape_params: N X number of shape parameters
            expression_params: N X number of expression parameters
            pose_params: N X number of pose parameters (6)
        return:d
            vertices: N X V X 3
            landmarks: N X number of landmarks X 3
        """
        batch_size = shape_params.shape[0]

        if global_rot is not None:
            global_rot_mat = batch_rodrigues(rot_vecs=global_rot)
        else:
            global_rot_mat = None

        if eye_pose_params is None:
            eye_pose_params = self.eye_pose.expand(batch_size, -1).to(
                shape_params.device
            )
        betas = torch.cat([shape_params, expression_params], dim=1)

        if pose_params.shape[1] == 6:
            full_pose = torch.cat(
                [
                    pose_params[:, :3],  # GLobal Rotation
                    self.neck_pose.expand(batch_size, -1).to(
                        shape_params.device
                    ),  # Neck pose
                    pose_params[:, 3:],  # Jaw pose
                    eye_pose_params,  # Eye pose
                ],
                dim=1,
            )
        elif pose_params.shape[1] == 9:
            full_pose = torch.cat(
                [
                    pose_params,
                    eye_pose_params,  # Eye pose
                ],
                dim=1,
            )
        else:
            raise ValueError
        template_vertices = self.v_template.unsqueeze(0).expand(batch_size, -1, -1)

        if per_vertex_def is not None:
            template_vertices = template_vertices + per_vertex_def

        vertices, *_ = lbs(
            betas,
            full_pose,
            template_vertices,
            self.shapedirs,
            self.posedirs,
            self.J_regressor,
            self.parents,
            self.lbs_weights,
            dtype=self.dtype,
        )

        if global_rot_mat is not None:
            assert len(vertices.shape) == 3
            # num_batches = vertices.shape[0]
            # _vertices_list = []

            # for i in range(num_batches):
            #    curr_rot_mat = global_rot_mat[i][None]
            #    curr_rot_mat = einops.repeat(
            #        curr_rot_mat, "b h w -> (repeat b) h w", repeat=vertices.shape[1]
            #    )
            #    curr_vertices = einops.rearrange(vertices[i], "n d -> n 1 d")
            #    _vertices = torch.bmm(curr_vertices, curr_rot_mat)
            #    _vertices_list.append(_vertices[:, 0, :])
            # vertices = torch.stack(_vertices_list, dim=0)

            global_rot_mat = einops.rearrange(global_rot_mat, "b h w -> b w h")
            assert len(vertices.shape) == 3
            assert len(global_rot_mat.shape) == 3
            assert vertices.shape[0] == global_rot_mat.shape[0]
            vertices = torch.matmul(vertices, global_rot_mat)

        if translation is None:
            translation = torch.zeros((1, vertices.shape[1], 3)).to(vertices.device)
        else:
            translation = translation[:, None, ...].repeat(1, vertices.shape[1], 1)

        vertices = vertices + translation

        lmk_faces_idx = self.lmk_faces_idx.unsqueeze(dim=0).expand(batch_size, -1)
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(dim=0).expand(
            batch_size, -1, -1
        )

        dyn_lmk_faces_idx, dyn_lmk_bary_coords = self._find_dynamic_lmk_idx_and_bcoords(
            full_pose,
            self.dynamic_lmk_faces_idx,
            self.dynamic_lmk_bary_coords,
            self.neck_kin_chain,
            dtype=self.dtype,
        )
        lmk_faces_idx = torch.cat([dyn_lmk_faces_idx, lmk_faces_idx], 1)
        lmk_bary_coords = torch.cat([dyn_lmk_bary_coords, lmk_bary_coords], 1)

        landmarks2d = vertices2landmarks(
            vertices, self.faces_tensor, lmk_faces_idx, lmk_bary_coords
        )
        bz = vertices.shape[0]
        landmarks3d = vertices2landmarks(
            vertices,
            self.faces_tensor,
            self.full_lmk_faces_idx.repeat(bz, 1),
            self.full_lmk_bary_coords.repeat(bz, 1, 1),
        )

        return vertices, landmarks2d, landmarks3d


class FLAMETex(nn.Module):
    """
    FLAME texture:
    https://github.com/TimoBolkart/TF_FLAME/blob/ade0ab152300ec5f0e8555d6765411555c5ed43d/sample_texture.py#L64
    FLAME texture converted from BFM:
    https://github.com/TimoBolkart/BFM_to_FLAME
    """

    def __init__(self, config):
        super(FLAMETex, self).__init__()

        if config.tex_type == "BFM":
            mu_key = "MU"
            pc_key = "PC"
            n_pc = 199
            tex_path = config.tex_path
            tex_space = np.load(tex_path)
            texture_mean = tex_space[mu_key].reshape(1, -1)
            texture_basis = tex_space[pc_key].reshape(-1, n_pc)
        elif config.tex_type == "AlbedoMM":
            mu_key = "MU"
            pc_key = "PC"
            n_pc = 145
            tex_path = config.tex_path
            tex_space = np.load(tex_path)
            texture_mean = tex_space[mu_key].reshape(1, -1)
            texture_basis = tex_space[pc_key].reshape(-1, n_pc)
        elif config.tex_type == "FLAME":
            mu_key = "mean"
            pc_key = "tex_dir"
            n_pc = 200
            tex_path = config.flame_tex_path
            tex_space = np.load(tex_path)
            texture_mean = tex_space[mu_key].reshape(1, -1) / 255.0
            texture_basis = tex_space[pc_key].reshape(-1, n_pc) / 255.0
        else:
            print("texture type ", config.tex_type, "not exist!")
            raise NotImplementedError

        n_tex = config.n_tex
        num_components = texture_basis.shape[1]
        texture_mean = torch.from_numpy(texture_mean).float()[None, ...]
        texture_basis = torch.from_numpy(texture_basis[:, :n_tex]).float()[None, ...]
        self.register_buffer("texture_mean", texture_mean)
        self.register_buffer("texture_basis", texture_basis)
        self.config_tex_type = config.tex_type

    def forward(self, texcode):
        """
        texcode: [batchsize, n_tex]
        texture: [bz, 3, 256, 256], range: 0-1
        """

        if self.config_tex_type == "AlbedoMM":
            texture = torch.clamp(
                self.texture_mean + (self.texture_basis * texcode[:, None, :]).sum(-1),
                0.0,
                1.0,
            )
            texture = torch.pow((texture + 1e-8), 1 / 2.2)
            texture = texture.reshape(texcode.shape[0], 512, 512, 3).permute(0, 3, 1, 2)
            texture = F.interpolate(texture, [256, 256])
            texture = texture[:, [2, 1, 0], :, :]
        else:
            texture = self.texture_mean + (
                self.texture_basis * texcode[:, None, :]
            ).sum(-1)
            texture = texture.reshape(texcode.shape[0], 512, 512, 3).permute(0, 3, 1, 2)
            texture = F.interpolate(texture, [256, 256])
            texture = texture[:, [2, 1, 0], :, :]

        return texture
