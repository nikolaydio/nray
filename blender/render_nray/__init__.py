bl_info = {
    "name": "NRay Renderer",
    "author": "NRay",
    "version": (0, 1, 0),
    "blender": (2, 65, 9),
    "location": "-",
    "description": "Basic PBR",
    "category": "Render",
}


if "bpy" in locals():
    import importlib
    importlib.reload(panel)

from . import panel

import inspect

import bpy
from bpy.props import (StringProperty,
                       BoolProperty,
                       IntProperty,
                       FloatVectorProperty,
                       FloatProperty,
                       CollectionProperty,
                       PointerProperty)

class RenderNRaySettingsScene(bpy.types.PropertyGroup):
    allowed = BoolProperty(default=True)
    samples = IntProperty("Samples", default=64, min=1, max=4192)
    resolution_x = IntProperty("Resolution X", default=800, min=1, max=1920)
    resolution_y = IntProperty("Resolution Y", default=600, min=1, max=1080)

class RenderNRaySettingsSetting(bpy.types.PropertyGroup):
    strid = StringProperty(default="")
    copy = BoolProperty(default=False)

class RenderNRaySettingsMaterial(bpy.types.PropertyGroup):
    @classmethod
    def register(cls):
        bpy.types.Material.nray = PointerProperty(
                name="NRay Material Settings",
                description="Nray material settings",
                type=cls,
                )
        cls.albedo = FloatVectorProperty(subtype="COLOR")
        cls.metalness = FloatProperty(min=0.0, max=1.0)
        cls.roughness = FloatProperty(min=0.0, max=1.0)
        cls.emissiveness = FloatProperty(min=0.0, max=1.0)
    @classmethod
    def unregister(cls):
        del bpy.types.Material.nray

class RenderNRaySettings(bpy.types.PropertyGroup):
    samples = IntProperty()


def write_matrix(fw, mat):
    for i in range(16):
        fw("%s " % mat[i//4][i%4])
    fw("\n")
import bmesh
from subprocess import Popen, PIPE, STDOUT

class NRayRenderer(bpy.types.RenderEngine):
    # These three members are used by blender to set up the
    # RenderEngine; define its internal name, visible name and capabilities.
    bl_idname = 'nray_renderer'
    bl_label = 'N Ray renderer'
    bl_use_preview = True

    # This is the only method called by blender, in this example
    # we use it to detect preview rendering and call the implementation
    # in another method.
    def render(self, scene):
        self.scale = scene.render.resolution_percentage / 100
        self.size_x = int(scene.render.resolution_x * self.scale)
        self.size_y = int(scene.render.resolution_y * self.scale)

        print("==============================SCENE================================")
        print(inspect.getmembers(scene))
        print("==============================CAMERA================================")
        print(inspect.getmembers(scene.camera))
        print("==============================OBJECTS================================")
        print(inspect.getmembers(scene.objects))

        self.render_scene(scene)

    def write_config(self, fw, scene):
        fw("NRAY-INTERNAL\n")
        fw("%s\n" % scene.nray.samples)
        fw("%s %s\n" % (self.size_x, self.size_y))

        write_matrix(fw, scene.camera.matrix_world)
        fw("%s\n" % 60)

        #start materials
        print("==============================MATERIALS================================")
        print(inspect.getmembers(bpy.data.materials))

        mats = bpy.data.materials
        mat_idxmap = {}
        fw("%s\n" % len(mats))
        for m in mats:
            fw("%s %s %s %s %s %s\n" % (m.nray.albedo[0], m.nray.albedo[1], m.nray.albedo[2], m.nray.metalness, m.nray.roughness, m.nray.emissiveness))
            mat_idxmap[m.name] = len(mat_idxmap)

        objs = scene.objects
        print("==============================Objects================================")
        print(inspect.getmembers(objs))
        #write meshes
        count = 0
        for i in objs:
            if i.type == "MESH":
                count += 1
        fw("%s\n" % count)

        for i in objs:
            if i.type == "MESH":
                mesh = i.to_mesh(scene, True, "RENDER")
                bm = bmesh.new()   # create an empty BMesh
                bm.from_mesh(mesh)   # fill it in from a Mesh
                bmesh.ops.triangulate(bm, faces=bm.faces)

                fw("%s\n" % len(bm.faces))
                for face in bm.faces:
                    for vert in face.verts:
                        fw("%s %s %s\n" % (vert.co.x, vert.co.y, vert.co.z));

                bm.free()
                del bm

        #write objs and transforms
        fw("%s\n" % count)
        ct = 0
        for i in objs:
            if i.type == "MESH":
                fw("%s %s " % (mat_idxmap[i.active_material.name], ct));
                write_matrix(fw, i.matrix_world)
                ct += 1

    # In this example, we fill the full renders with a flat blue color.
    def render_scene(self, scene):
        pixel_count = self.size_x * self.size_y
        self.p = Popen(['./nray', 'headless'], stdout=PIPE, stdin=PIPE, stderr=PIPE)
        out = open("Output.txt", "w")
        self.write_config(lambda s: out.write(s), scene)
        out.close()
        self.write_config(lambda s: self.p.stdin.write(bytes(s, 'UTF-8')), scene)
        self.p.stdin.flush()


        # Here we write the pixel values to the RenderResult
        result = self.begin_result(0, 0, self.size_x, self.size_y)
        layer = result.layers[0]

        re = [[0.0, 0.0, 0.0, 1.0] for i in range(pixel_count)]
        print("Starting plotting ", layer.rect)
        for u in range(scene.nray.samples):
            for i in range(pixel_count):
                l = self.p.stdout.readline().decode("utf-8")
                pp = l.split()

                re[i][0] = int(pp[0]) / 255.0
                re[i][1] = int(pp[1]) / 255.0
                re[i][2] = int(pp[2]) / 255.0
            layer.rect = re
            self.update_result(result)
            self.update_progress(u / scene.nray.samples)
            if self.test_break():
                break
            #print("YAY")
        self.end_result(result)
        self.p.kill()

def register():
    # Register properties.
    bpy.utils.register_class(RenderNRaySettingsScene)
    bpy.utils.register_class(RenderNRaySettingsSetting)
    bpy.utils.register_class(RenderNRaySettings)
    bpy.utils.register_class(RenderNRaySettingsMaterial);

    bpy.types.Scene.nray = PointerProperty(type=RenderNRaySettingsScene)
    bpy.types.Material.nray = PointerProperty(type=RenderNRaySettingsMaterial)
    bpy.utils.register_module(__name__)


def unregister():
    # Unregister properties.
    bpy.utils.unregister_class(RenderNRaySettingsScene)
    bpy.utils.unregister_class(RenderNRaySettingsSetting)
    bpy.utils.unregister_class(RenderNRaySettings)
    bpy.utils.unregister_class(RenderNRaySettingsMaterial);
    del bpy.types.Scene.nray
    del bpy.types.Material.nray

    bpy.utils.unregister_module(__name__)


if __name__ == "__main__":
    register()
