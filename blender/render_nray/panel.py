# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# ##### END GPL LICENSE BLOCK #####

# <pep8 compliant>

import bpy


class RENDER_PT_render_settings(bpy.types.Panel):
    bl_label = "Render Settings"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "render"
    bl_options = {'DEFAULT_CLOSED'}
    COMPAT_ENGINES = {'nray_renderer'}

    samples = bpy.props.IntProperty(name="Steps", default=64, min=1, max=4192)

    def draw(self, context):
        layout = self.layout
        nray = context.scene.nray
        row = layout.row(align=True)
        row.prop(nray, "samples")
        row = layout.row(align=True)
        row.prop(nray, "resolution_x")
        row.prop(nray, "resolution_y")


class RENDER_PT_context_material(bpy.types.Panel):
    bl_label = "Material List"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "material"
    bl_options = {'DEFAULT_CLOSED'}
    COMPAT_ENGINES = {'nray_renderer'}

    def draw(self, context):
        layout = self.layout

        mat = context.material
        ob = context.object
        slot = context.material_slot
        space = context.space_data

        if ob:
            row = layout.row()

            row.template_list("MATERIAL_UL_matslots", "", ob, "material_slots", ob, "active_material_index", rows=1)

            col = row.column(align=True)
            col.operator("object.material_slot_add", icon='ZOOMIN', text="")
            col.operator("object.material_slot_remove", icon='ZOOMOUT', text="")

            col.menu("MATERIAL_MT_specials", icon='DOWNARROW_HLT', text="")

            if ob.mode == 'EDIT':
                row = layout.row(align=True)
                row.operator("object.material_slot_assign", text="Assign")
                row.operator("object.material_slot_select", text="Select")
                row.operator("object.material_slot_deselect", text="Deselect")

        split = layout.split(percentage=0.65)
        if ob:
            split.template_ID(ob, "active_material", new="material.new")
            row = split.row()

            if slot:
                row.prop(slot, "link", text="")
            else:
                row.label()
        elif mat:
            split.template_ID(space, "pin_id")
            split.separator()

class NRayMaterial_PT_surface(bpy.types.Panel):
    bl_label = "Surface"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "material"
    bl_options = {'DEFAULT_CLOSED'}
    COMPAT_ENGINES = {'nray_renderer'}

    def draw(self, context):
        layout = self.layout
        if context.material == None:
            return
        mat = context.material.nray
        layout.prop(mat, "albedo")
        layout.prop(mat, "metalness")
        layout.prop(mat, "roughness")
        layout.prop(mat, "emissiveness")
