from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import glfw
import numpy as np
import OpenGL.GL as GL

from libs import transform as T
from libs.shader import Shader

from .mesh_loader import MaterialData, MeshData


@dataclass
class CameraParams:
    eye: np.ndarray
    target: np.ndarray
    up: np.ndarray
    fovy: float
    near: float
    far: float


@dataclass
class LightParams:
    sun_direction: np.ndarray
    sun_color: np.ndarray
    fill_position: np.ndarray
    fill_color: np.ndarray
    sky_color: np.ndarray
    ambient_strength: float = 0.28
    sun_intensity: float = 1.0
    fill_intensity: float = 1.0
    shadow_strength: float = 0.65
    shadow_enabled: bool = True
    street_light_positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))
    street_light_color: np.ndarray = field(default_factory=lambda: np.asarray([1.0, 0.90, 0.68], dtype=np.float32))
    street_light_intensity: float = 0.0


@dataclass
class SceneInstance:
    mesh_key: str
    class_id: int
    class_name: str
    instance_id: int
    model_matrix: np.ndarray
    use_alpha_key: bool = False
    wheel_spin: float = 0.0
    wheel_slide: float = 0.0


@dataclass
class RenderResult:
    rgb: np.ndarray  # (H, W, 3) uint8
    depth_linear: np.ndarray  # (H, W) float32
    instance_map: np.ndarray  # (H, W) int32


@dataclass
class _GpuSubmesh:
    material_id: int
    first_index: int
    index_count: int


@dataclass
class _GpuMesh:
    key: str
    vao: int
    vbo: int
    ebo: int
    index_count: int
    submeshes: List[_GpuSubmesh]
    materials: List[MaterialData]
    bbox_min: np.ndarray
    bbox_max: np.ndarray


class GLDatasetRenderer:
    def __init__(self, width: int, height: int, shaders_dir: Path) -> None:
        self.width = int(width)
        self.height = int(height)
        self.shaders_dir = Path(shaders_dir).resolve()

        self.window = None
        self.rgb_shader: Optional[Shader] = None
        self.id_shader: Optional[Shader] = None
        self.shadow_shader: Optional[Shader] = None

        self.rgb_fbo = 0
        self.rgb_color_tex = 0
        self.rgb_depth_rbo = 0

        self.id_fbo = 0
        self.id_color_tex = 0
        self.id_depth_rbo = 0

        self.shadow_map_size = 2048
        self.shadow_fbo = 0
        self.shadow_depth_tex = 0

        self.meshes: Dict[str, _GpuMesh] = {}
        self.texture_cache: Dict[str, int] = {}

        self._init_gl()
        self._init_programs()
        self._init_framebuffers()

    def close(self) -> None:
        # Release shader objects while GL context is still alive.
        if self.rgb_shader is not None:
            shader = self.rgb_shader
            self.rgb_shader = None
            del shader
        if self.id_shader is not None:
            shader = self.id_shader
            self.id_shader = None
            del shader
        if self.shadow_shader is not None:
            shader = self.shadow_shader
            self.shadow_shader = None
            del shader

        for mesh in self.meshes.values():
            GL.glDeleteVertexArrays(1, [mesh.vao])
            GL.glDeleteBuffers(1, [mesh.vbo])
            GL.glDeleteBuffers(1, [mesh.ebo])
        self.meshes.clear()

        for tex in self.texture_cache.values():
            GL.glDeleteTextures(1, [tex])
        self.texture_cache.clear()

        if self.rgb_color_tex:
            GL.glDeleteTextures(1, [self.rgb_color_tex])
        if self.rgb_depth_rbo:
            GL.glDeleteRenderbuffers(1, [self.rgb_depth_rbo])
        if self.rgb_fbo:
            GL.glDeleteFramebuffers(1, [self.rgb_fbo])

        if self.id_color_tex:
            GL.glDeleteTextures(1, [self.id_color_tex])
        if self.id_depth_rbo:
            GL.glDeleteRenderbuffers(1, [self.id_depth_rbo])
        if self.id_fbo:
            GL.glDeleteFramebuffers(1, [self.id_fbo])

        if self.shadow_depth_tex:
            GL.glDeleteTextures(1, [self.shadow_depth_tex])
        if self.shadow_fbo:
            GL.glDeleteFramebuffers(1, [self.shadow_fbo])

        if self.window is not None:
            glfw.destroy_window(self.window)
            self.window = None
        if glfw.get_current_context() is not None:
            glfw.make_context_current(None)
        glfw.terminate()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _init_gl(self) -> None:
        if not glfw.init():
            raise RuntimeError("glfw.init() failed")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        self.window = glfw.create_window(self.width, self.height, "btl2_offscreen", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create hidden OpenGL window")
        glfw.make_context_current(self.window)

        GL.glViewport(0, 0, self.width, self.height)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LESS)

    def _init_programs(self) -> None:
        self.rgb_shader = Shader(
            str(self.shaders_dir / "rgb.vert"),
            str(self.shaders_dir / "rgb.frag"),
        )
        self.id_shader = Shader(
            str(self.shaders_dir / "id.vert"),
            str(self.shaders_dir / "id.frag"),
        )
        self.shadow_shader = Shader(
            str(self.shaders_dir / "shadow.vert"),
            str(self.shaders_dir / "shadow.frag"),
        )

    @staticmethod
    def _check_fbo_complete(name: str) -> None:
        status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        if status != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"{name} incomplete. OpenGL status={status}")

    def _init_framebuffers(self) -> None:
        # RGB framebuffer
        self.rgb_fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.rgb_fbo)

        self.rgb_color_tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.rgb_color_tex)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGB8,
            self.width,
            self.height,
            0,
            GL.GL_RGB,
            GL.GL_UNSIGNED_BYTE,
            None,
        )
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self.rgb_color_tex, 0
        )

        self.rgb_depth_rbo = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self.rgb_depth_rbo)
        GL.glRenderbufferStorage(
            GL.GL_RENDERBUFFER,
            GL.GL_DEPTH_COMPONENT24,
            self.width,
            self.height,
        )
        GL.glFramebufferRenderbuffer(
            GL.GL_FRAMEBUFFER,
            GL.GL_DEPTH_ATTACHMENT,
            GL.GL_RENDERBUFFER,
            self.rgb_depth_rbo,
        )
        self._check_fbo_complete("rgb_fbo")

        # ID framebuffer
        self.id_fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.id_fbo)

        self.id_color_tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.id_color_tex)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGB8,
            self.width,
            self.height,
            0,
            GL.GL_RGB,
            GL.GL_UNSIGNED_BYTE,
            None,
        )
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self.id_color_tex, 0
        )

        self.id_depth_rbo = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self.id_depth_rbo)
        GL.glRenderbufferStorage(
            GL.GL_RENDERBUFFER,
            GL.GL_DEPTH_COMPONENT24,
            self.width,
            self.height,
        )
        GL.glFramebufferRenderbuffer(
            GL.GL_FRAMEBUFFER,
            GL.GL_DEPTH_ATTACHMENT,
            GL.GL_RENDERBUFFER,
            self.id_depth_rbo,
        )
        self._check_fbo_complete("id_fbo")

        # Shadow framebuffer (depth-only)
        self.shadow_fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.shadow_fbo)

        self.shadow_depth_tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.shadow_depth_tex)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_DEPTH_COMPONENT32F,
            self.shadow_map_size,
            self.shadow_map_size,
            0,
            GL.GL_DEPTH_COMPONENT,
            GL.GL_FLOAT,
            None,
        )
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_BORDER)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_BORDER)
        GL.glTexParameterfv(
            GL.GL_TEXTURE_2D,
            GL.GL_TEXTURE_BORDER_COLOR,
            np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        )
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER,
            GL.GL_DEPTH_ATTACHMENT,
            GL.GL_TEXTURE_2D,
            self.shadow_depth_tex,
            0,
        )
        GL.glDrawBuffer(GL.GL_NONE)
        GL.glReadBuffer(GL.GL_NONE)
        self._check_fbo_complete("shadow_fbo")

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def _upload_texture(self, image_path: Path, prefer_alpha_mask: bool = False) -> int:
        image_path = image_path.resolve()
        cache_key = f"{str(image_path).lower()}|alpha={1 if prefer_alpha_mask else 0}"
        if cache_key in self.texture_cache:
            return self.texture_cache[cache_key]

        # `cv2.imread` fails on some Windows unicode paths; use imdecode for robustness.
        raw = np.fromfile(str(image_path), dtype=np.uint8)
        image = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Could not load texture: {image_path}")

        if image.ndim == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            fmt = GL.GL_RGB
            internal_fmt = GL.GL_RGB8
        elif image.shape[2] == 4:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            fmt = GL.GL_RGBA
            internal_fmt = GL.GL_RGBA8
        else:
            rgb3 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if prefer_alpha_mask:
                # Build alpha from estimated background color on image borders.
                h, w = rgb3.shape[:2]
                border_parts = [
                    rgb3[0, :, :],
                    rgb3[h - 1, :, :],
                    rgb3[:, 0, :],
                    rgb3[:, w - 1, :],
                ]
                border = np.concatenate(border_parts, axis=0).astype(np.float32)
                bg_rgb = np.median(border, axis=0)

                rgb_f = rgb3.astype(np.float32)
                color_dist = np.linalg.norm(rgb_f - bg_rgb.reshape(1, 1, 3), axis=2)
                mean_rgb = rgb_f.mean(axis=2)
                std_rgb = rgb_f.std(axis=2)

                # Foreground survives when:
                # - sufficiently different from border background color, or
                # - has strong local color variance, or
                # - is distinctly dark (common for hair/tires/clothes on white backgrounds).
                fg_mask = (color_dist > 30.0) | (std_rgb > 7.0) | (mean_rgb < 185.0)
                alpha = (fg_mask.astype(np.uint8) * 255)
                rgb = np.dstack([rgb3, alpha]).astype(np.uint8)
                fmt = GL.GL_RGBA
                internal_fmt = GL.GL_RGBA8
            else:
                rgb = rgb3
                fmt = GL.GL_RGB
                internal_fmt = GL.GL_RGB8

        # OBJ UVs are defined in a bottom-left convention while image decoders
        # return top-left origin. Flip once here to keep textures upright.
        rgb = np.ascontiguousarray(np.flipud(rgb))

        tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            internal_fmt,
            rgb.shape[1],
            rgb.shape[0],
            0,
            fmt,
            GL.GL_UNSIGNED_BYTE,
            rgb,
        )
        GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)

        self.texture_cache[cache_key] = tex
        return tex

    def register_mesh(self, key: str, mesh: MeshData) -> None:
        if key in self.meshes:
            return

        tri_order = np.argsort(mesh.tri_material_ids, kind="stable")
        triangles = mesh.triangles[tri_order]
        tri_mats = mesh.tri_material_ids[tri_order]

        flat_indices = triangles.reshape(-1).astype(np.uint32)
        interleaved = np.hstack([mesh.vertices, mesh.normals, mesh.uvs]).astype(np.float32)

        vao = GL.glGenVertexArrays(1)
        vbo = GL.glGenBuffers(1)
        ebo = GL.glGenBuffers(1)

        GL.glBindVertexArray(vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, interleaved.nbytes, interleaved, GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, ebo)
        GL.glBufferData(
            GL.GL_ELEMENT_ARRAY_BUFFER,
            flat_indices.nbytes,
            flat_indices,
            GL.GL_STATIC_DRAW,
        )

        stride = 8 * 4
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(3 * 4))
        GL.glEnableVertexAttribArray(2)
        GL.glVertexAttribPointer(2, 2, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(6 * 4))
        GL.glBindVertexArray(0)

        submeshes: List[_GpuSubmesh] = []
        start = 0
        total_tri = tri_mats.shape[0]
        while start < total_tri:
            mat_id = int(tri_mats[start])
            end = start + 1
            while end < total_tri and int(tri_mats[end]) == mat_id:
                end += 1
            submeshes.append(
                _GpuSubmesh(material_id=mat_id, first_index=start * 3, index_count=(end - start) * 3)
            )
            start = end

        self.meshes[key] = _GpuMesh(
            key=key,
            vao=vao,
            vbo=vbo,
            ebo=ebo,
            index_count=int(flat_indices.shape[0]),
            submeshes=submeshes,
            materials=mesh.materials,
            bbox_min=mesh.bbox_min.copy(),
            bbox_max=mesh.bbox_max.copy(),
        )

    @staticmethod
    def encode_instance_id(instance_id: int) -> np.ndarray:
        if instance_id < 0 or instance_id > 0xFFFFFF:
            raise ValueError(f"instance_id out of range: {instance_id}")
        r = (instance_id & 255) / 255.0
        g = ((instance_id >> 8) & 255) / 255.0
        b = ((instance_id >> 16) & 255) / 255.0
        return np.asarray([r, g, b], dtype=np.float32)

    @staticmethod
    def decode_instance_map(rgb_map: np.ndarray) -> np.ndarray:
        rgb_map = rgb_map.astype(np.int32)
        return rgb_map[:, :, 0] + (rgb_map[:, :, 1] << 8) + (rgb_map[:, :, 2] << 16)

    def _upload_mat4(self, program: int, name: str, matrix: np.ndarray) -> None:
        loc = GL.glGetUniformLocation(program, name)
        GL.glUniformMatrix4fv(loc, 1, True, matrix.astype(np.float32))

    def _upload_vec3(self, program: int, name: str, vec3: np.ndarray) -> None:
        loc = GL.glGetUniformLocation(program, name)
        GL.glUniform3fv(loc, 1, np.asarray(vec3, dtype=np.float32))

    def _upload_vec2(self, program: int, name: str, vec2: np.ndarray) -> None:
        loc = GL.glGetUniformLocation(program, name)
        GL.glUniform2fv(loc, 1, np.asarray(vec2, dtype=np.float32))

    def _upload_float(self, program: int, name: str, value: float) -> None:
        loc = GL.glGetUniformLocation(program, name)
        GL.glUniform1f(loc, float(value))

    def _upload_int(self, program: int, name: str, value: int) -> None:
        loc = GL.glGetUniformLocation(program, name)
        GL.glUniform1i(loc, int(value))

    def _upload_vec3_array(self, program: int, name: str, values: np.ndarray) -> None:
        loc = GL.glGetUniformLocation(program, name)
        arr = np.asarray(values, dtype=np.float32)
        if arr.size == 0:
            return
        GL.glUniform3fv(loc, arr.shape[0], arr)

    def _upload_street_lights(self, program: int, light: LightParams, max_lights: int = 16) -> None:
        positions = np.asarray(light.street_light_positions, dtype=np.float32).reshape(-1, 3)
        if positions.size == 0 or float(light.street_light_intensity) <= 1e-5:
            self._upload_int(program, "street_light_count", 0)
            self._upload_float(program, "street_light_intensity", 0.0)
            self._upload_vec3(program, "street_light_color", np.asarray(light.street_light_color, dtype=np.float32))
            return

        clipped = positions[: max(1, int(max_lights))]
        self._upload_int(program, "street_light_count", int(clipped.shape[0]))
        self._upload_vec3(program, "street_light_color", np.asarray(light.street_light_color, dtype=np.float32))
        self._upload_float(program, "street_light_intensity", float(max(light.street_light_intensity, 0.0)))
        self._upload_vec3_array(program, "street_light_positions", clipped)

    @staticmethod
    def _normalize3(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        arr = np.asarray(vec, dtype=np.float32)
        n = float(np.linalg.norm(arr))
        if n < 1e-8:
            return np.asarray(fallback, dtype=np.float32)
        return (arr / n).astype(np.float32)

    @staticmethod
    def _bbox_corners(bbox_min: np.ndarray, bbox_max: np.ndarray) -> np.ndarray:
        bmin = np.asarray(bbox_min, dtype=np.float32)
        bmax = np.asarray(bbox_max, dtype=np.float32)
        return np.asarray(
            [
                [bmin[0], bmin[1], bmin[2]],
                [bmax[0], bmin[1], bmin[2]],
                [bmin[0], bmax[1], bmin[2]],
                [bmax[0], bmax[1], bmin[2]],
                [bmin[0], bmin[1], bmax[2]],
                [bmax[0], bmin[1], bmax[2]],
                [bmin[0], bmax[1], bmax[2]],
                [bmax[0], bmax[1], bmax[2]],
            ],
            dtype=np.float32,
        )

    def _compute_scene_bounds(self, instances: List[SceneInstance]) -> Tuple[np.ndarray, np.ndarray]:
        all_points: List[np.ndarray] = []
        for inst in instances:
            mesh = self.meshes.get(inst.mesh_key)
            if mesh is None:
                continue
            corners = self._bbox_corners(mesh.bbox_min, mesh.bbox_max)
            corners_h = np.hstack([corners, np.ones((corners.shape[0], 1), dtype=np.float32)])
            world = (inst.model_matrix @ corners_h.T).T[:, :3]
            all_points.append(world)

        if not all_points:
            return (
                np.asarray([-30.0, -2.0, -90.0], dtype=np.float32),
                np.asarray([30.0, 20.0, 20.0], dtype=np.float32),
            )

        stacked = np.vstack(all_points)
        return stacked.min(axis=0).astype(np.float32), stacked.max(axis=0).astype(np.float32)

    def _compute_light_space_matrix(self, light: LightParams, instances: List[SceneInstance]) -> np.ndarray:
        scene_min, scene_max = self._compute_scene_bounds(instances)
        center = ((scene_min + scene_max) * 0.5).astype(np.float32)
        diag = float(np.linalg.norm(scene_max - scene_min))
        diag = max(diag, 1.0)

        sun_dir = self._normalize3(light.sun_direction, fallback=np.asarray([0.35, -1.0, -0.2], dtype=np.float32))
        light_distance = max(35.0, diag * 1.55)
        light_eye = (center - sun_dir * light_distance).astype(np.float32)

        up = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(self._normalize3(sun_dir, up), up))) > 0.95:
            up = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)

        light_view = T.lookat(light_eye, center, up).astype(np.float32)

        corners = self._bbox_corners(scene_min, scene_max)
        corners_h = np.hstack([corners, np.ones((corners.shape[0], 1), dtype=np.float32)])
        light_space_pts = (light_view @ corners_h.T).T[:, :3]

        mins = light_space_pts.min(axis=0)
        maxs = light_space_pts.max(axis=0)
        margin_xy = max(2.0, diag * 0.06)
        margin_z = max(8.0, diag * 0.22)

        left = float(mins[0] - margin_xy)
        right = float(maxs[0] + margin_xy)
        bottom = float(mins[1] - margin_xy)
        top = float(maxs[1] + margin_xy)
        near = float(mins[2] - margin_z)
        far = float(maxs[2] + margin_z)

        if right <= left + 1e-4:
            right = left + 1.0
        if top <= bottom + 1e-4:
            top = bottom + 1.0
        if far <= near + 1e-3:
            far = near + 10.0

        light_proj = T.ortho(left, right, bottom, top, near, far).astype(np.float32)
        return (light_proj @ light_view).astype(np.float32)

    def _draw_shadow_instance(self, inst: SceneInstance, mesh: _GpuMesh) -> None:
        assert self.shadow_shader is not None
        program = self.shadow_shader.render_idx
        self._upload_mat4(program, "model", inst.model_matrix)
        self._upload_int(program, "use_alpha_key", 1 if inst.use_alpha_key else 0)

        GL.glBindVertexArray(mesh.vao)
        for sub in mesh.submeshes:
            mat = mesh.materials[sub.material_id]
            uv_offset = np.asarray([0.0, 0.0], dtype=np.float32)
            if self._is_wheel_like_material(mat):
                uv_offset = np.asarray(
                    [
                        float(inst.wheel_spin + inst.wheel_slide),
                        float(inst.wheel_slide * 0.25),
                    ],
                    dtype=np.float32,
                )
            self._upload_vec2(program, "uv_offset", uv_offset)

            if mat.texture_path is not None and mat.texture_path.exists():
                tex = self._upload_texture(mat.texture_path, prefer_alpha_mask=inst.use_alpha_key)
                GL.glActiveTexture(GL.GL_TEXTURE0)
                GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
                self._upload_int(program, "tex0", 0)
                self._upload_int(program, "use_texture", 1)
            else:
                self._upload_int(program, "use_texture", 0)

            GL.glDrawElements(
                GL.GL_TRIANGLES,
                sub.index_count,
                GL.GL_UNSIGNED_INT,
                ctypes.c_void_p(sub.first_index * 4),
            )
        GL.glBindVertexArray(0)

    def _render_shadow_map(self, light_space_matrix: np.ndarray, instances: List[SceneInstance]) -> None:
        assert self.shadow_shader is not None

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.shadow_fbo)
        GL.glViewport(0, 0, self.shadow_map_size, self.shadow_map_size)
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_CULL_FACE)
        GL.glCullFace(GL.GL_FRONT)

        GL.glUseProgram(self.shadow_shader.render_idx)
        self._upload_mat4(self.shadow_shader.render_idx, "light_space_matrix", light_space_matrix)
        for inst in instances:
            mesh = self.meshes[inst.mesh_key]
            self._draw_shadow_instance(inst, mesh)

        GL.glCullFace(GL.GL_BACK)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    @staticmethod
    def _is_wheel_like_material(mat: MaterialData) -> bool:
        tags = ("wheel", "tire", "tyre", "rim", "ban", "lop", "brake")
        haystacks = [mat.name.lower()]
        if mat.texture_path is not None:
            haystacks.append(mat.texture_path.stem.lower())
            haystacks.append(mat.texture_path.name.lower())
        joined = " ".join(haystacks)
        return any(tag in joined for tag in tags)

    def _draw_rgb_instance(self, inst: SceneInstance, mesh: _GpuMesh) -> None:
        assert self.rgb_shader is not None
        program = self.rgb_shader.render_idx
        self._upload_mat4(program, "model", inst.model_matrix)
        self._upload_int(program, "use_alpha_key", 1 if inst.use_alpha_key else 0)

        GL.glBindVertexArray(mesh.vao)
        for sub in mesh.submeshes:
            mat = mesh.materials[sub.material_id]
            self._upload_vec3(program, "material_kd", mat.kd)
            self._upload_vec3(program, "material_ks", mat.ks)
            self._upload_vec3(program, "material_ka", mat.ka)
            self._upload_float(program, "material_shininess", mat.shininess)
            uv_offset = np.asarray([0.0, 0.0], dtype=np.float32)
            if self._is_wheel_like_material(mat):
                # Approximate wheel rolling + lateral slip through texture scrolling.
                uv_offset = np.asarray(
                    [
                        float(inst.wheel_spin + inst.wheel_slide),
                        float(inst.wheel_slide * 0.25),
                    ],
                    dtype=np.float32,
                )
            self._upload_vec2(program, "uv_offset", uv_offset)

            if mat.texture_path is not None and mat.texture_path.exists():
                tex = self._upload_texture(mat.texture_path, prefer_alpha_mask=inst.use_alpha_key)
                GL.glActiveTexture(GL.GL_TEXTURE0)
                GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
                self._upload_int(program, "tex0", 0)
                self._upload_int(program, "use_texture", 1)
            else:
                self._upload_int(program, "use_texture", 0)

            GL.glDrawElements(
                GL.GL_TRIANGLES,
                sub.index_count,
                GL.GL_UNSIGNED_INT,
                ctypes.c_void_p(sub.first_index * 4),
            )
        GL.glBindVertexArray(0)

    def _draw_id_instance(self, inst: SceneInstance, mesh: _GpuMesh) -> None:
        assert self.id_shader is not None
        program = self.id_shader.render_idx
        self._upload_mat4(program, "model", inst.model_matrix)
        self._upload_vec3(program, "encoded_id_rgb", self.encode_instance_id(inst.instance_id))
        self._upload_int(program, "use_alpha_key", 1 if inst.use_alpha_key else 0)

        GL.glBindVertexArray(mesh.vao)
        for sub in mesh.submeshes:
            mat = mesh.materials[sub.material_id]
            if mat.texture_path is not None and mat.texture_path.exists():
                tex = self._upload_texture(mat.texture_path, prefer_alpha_mask=inst.use_alpha_key)
                GL.glActiveTexture(GL.GL_TEXTURE0)
                GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
                self._upload_int(program, "tex0", 0)
                self._upload_int(program, "use_texture", 1)
            else:
                self._upload_int(program, "use_texture", 0)

            GL.glDrawElements(
                GL.GL_TRIANGLES,
                sub.index_count,
                GL.GL_UNSIGNED_INT,
                ctypes.c_void_p(sub.first_index * 4),
            )
        GL.glBindVertexArray(0)

    @staticmethod
    def linearize_depth(depth_buffer: np.ndarray, near: float, far: float) -> np.ndarray:
        z_ndc = depth_buffer * 2.0 - 1.0
        linear = (2.0 * near * far) / (far + near - z_ndc * (far - near))
        linear = linear.astype(np.float32)
        linear[depth_buffer >= 1.0] = np.float32(far)
        return linear

    def render(
        self,
        camera: CameraParams,
        light: LightParams,
        instances: List[SceneInstance],
    ) -> RenderResult:
        assert self.rgb_shader is not None
        assert self.id_shader is not None
        assert self.shadow_shader is not None

        aspect = float(self.width) / float(self.height)
        view = T.lookat(camera.eye, camera.target, camera.up).astype(np.float32)
        projection = T.perspective(camera.fovy, aspect, camera.near, camera.far).astype(np.float32)

        light_space_matrix = np.identity(4, dtype=np.float32)
        if bool(light.shadow_enabled):
            light_space_matrix = self._compute_light_space_matrix(light, instances)
            self._render_shadow_map(light_space_matrix, instances)

        # RGB pass
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.rgb_fbo)
        GL.glViewport(0, 0, self.width, self.height)
        sky = np.clip(np.asarray(light.sky_color, dtype=np.float32), 0.0, 1.0)
        GL.glClearColor(float(sky[0]), float(sky[1]), float(sky[2]), 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_BLEND)

        GL.glUseProgram(self.rgb_shader.render_idx)
        self._upload_mat4(self.rgb_shader.render_idx, "view", view)
        self._upload_mat4(self.rgb_shader.render_idx, "projection", projection)
        self._upload_mat4(self.rgb_shader.render_idx, "light_space_matrix", light_space_matrix)
        self._upload_vec3(self.rgb_shader.render_idx, "camera_pos", camera.eye)
        self._upload_vec3(self.rgb_shader.render_idx, "sun_direction", light.sun_direction)
        self._upload_vec3(self.rgb_shader.render_idx, "sun_color", light.sun_color)
        self._upload_float(self.rgb_shader.render_idx, "sun_intensity", light.sun_intensity)
        self._upload_vec3(self.rgb_shader.render_idx, "fill_position", light.fill_position)
        self._upload_vec3(self.rgb_shader.render_idx, "fill_color", light.fill_color)
        self._upload_float(self.rgb_shader.render_idx, "fill_intensity", light.fill_intensity)
        self._upload_vec3(self.rgb_shader.render_idx, "sky_color", light.sky_color)
        self._upload_float(self.rgb_shader.render_idx, "ambient_strength", light.ambient_strength)
        self._upload_street_lights(self.rgb_shader.render_idx, light)
        self._upload_int(self.rgb_shader.render_idx, "shadow_enabled", 1 if light.shadow_enabled else 0)
        self._upload_float(self.rgb_shader.render_idx, "shadow_strength", light.shadow_strength)
        self._upload_float(self.rgb_shader.render_idx, "shadow_bias", 0.0014)
        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.shadow_depth_tex)
        self._upload_int(self.rgb_shader.render_idx, "shadow_map", 1)

        for inst in instances:
            mesh = self.meshes[inst.mesh_key]
            self._draw_rgb_instance(inst, mesh)

        rgb_raw = GL.glReadPixels(0, 0, self.width, self.height, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
        depth_raw = GL.glReadPixels(
            0,
            0,
            self.width,
            self.height,
            GL.GL_DEPTH_COMPONENT,
            GL.GL_FLOAT,
        )
        rgb = np.frombuffer(rgb_raw, dtype=np.uint8).reshape(self.height, self.width, 3)
        depth = np.frombuffer(depth_raw, dtype=np.float32).reshape(self.height, self.width)
        rgb = np.flipud(rgb)
        depth = np.flipud(depth)
        depth_linear = self.linearize_depth(depth, camera.near, camera.far)

        # ID pass
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.id_fbo)
        GL.glViewport(0, 0, self.width, self.height)
        GL.glClearColor(0.0, 0.0, 0.0, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glEnable(GL.GL_DEPTH_TEST)

        GL.glUseProgram(self.id_shader.render_idx)
        self._upload_mat4(self.id_shader.render_idx, "view", view)
        self._upload_mat4(self.id_shader.render_idx, "projection", projection)
        for inst in instances:
            mesh = self.meshes[inst.mesh_key]
            self._draw_id_instance(inst, mesh)

        id_raw = GL.glReadPixels(0, 0, self.width, self.height, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
        id_rgb = np.frombuffer(id_raw, dtype=np.uint8).reshape(self.height, self.width, 3)
        id_rgb = np.flipud(id_rgb)
        instance_map = self.decode_instance_map(id_rgb)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        return RenderResult(rgb=rgb, depth_linear=depth_linear, instance_map=instance_map.astype(np.int32))
