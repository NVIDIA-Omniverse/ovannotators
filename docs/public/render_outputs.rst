.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

ovrtx Render Outputs
====================

``OvrtxOutput`` extracts RenderVars by their `ovrtx
<https://github.com/NVIDIA-Omniverse/ovrtx>`_ ``sourceName``. Its default
``decode=True`` path maps image tensors through DLPack and decodes packed
annotator buffers before passing them to a writer. The complete, runtime-tested
camera catalog is maintained in the `ovrtx camera output documentation
<https://nvidia-omniverse.github.io/ovrtx/sensors/cameras/outputs.html>`_.

The renderer keys ``FrameOutput.render_vars`` by full RenderVar
prim path (``/Render/Camera/LdrColor``) and reports the AOV kind separately as
``source_name``. ``OvrtxOutput`` resolves the ``sourceName`` names in this page
against those paths, so render-var dictionaries and writer
``required_render_vars`` stay keyed by ``sourceName``. Pass a full prim path to
:meth:`~ovannotators.renderer_outputs.ovrtx.OvrtxOutput.extract` when one frame
authors two RenderVar prims that share a ``sourceName``.

The inventory below includes ovrtx 0.5 readback from one robot scene in minimal,
RT2, and path tracing modes. A single frame does not establish that a zero-size
buffer is unavailable in other scenes or configurations.

Image and Tensor Outputs
------------------------

``BasicWriter`` explicitly names all 124 ``sourceName`` values observed in that
capture. Six additional uint8 RGBA outputs (``LdrColorSD``, ``SimpleShadingSD``,
``DiffuseAlbedo``, ``SpecularAlbedo``, ``Roughness``, and
``DepthSensorPointCloudColor``) use PNG or JPEG according to ``image_format``.
Fifty-five additional float image outputs use EXR, including ``Pt*`` lighting
passes, motion, positions, and depth. Float32 input stays float32 in EXR, so
large values and depth sentinels are preserved with the default lossless
compression (lossy ``exr_compression`` schemes can change values). EXR contains raw values and
may need exposure or channel selection in an image viewer. JPEG is lossy; use
PNG for exact uint8 pixel values.

``Depth`` and ``DiffuseAlbedo`` are written as ``depth_native.exr`` and
``diffuse_albedo_native.png``, avoiding filename collisions with
the existing ``DepthSD`` and ``DiffuseAlbedoSD`` outputs. Integer ID images,
packed annotation records, delta buffers, and auxiliary data retain ``.npy``
output. Unknown names also fall back to ``.npy``. Empty arrays are skipped.

.. list-table::
   :header-rows: 1
   :widths: 34 24 42

   * - ``sourceName``
     - BasicWriter encoding
     - Notes
   * - ``LdrColor``
     - PNG/JPEG
     - Display-ready uint8 RGBA.
   * - ``HdrColor``
     - EXR
     - Linear RGBA; keeps the renderer's float32 (or float16) precision.
   * - ``NormalSD``
     - EXR plus colorized PNG
     - Working RT2 normal alternative to ``SmoothNormal``.
   * - ``DepthSD``
     - EXR or NPY
     - Python readback returned non-zero float32 depth. Deprecated by ovrtx
       0.5; prefer ``DistanceToImagePlaneSD`` or ``DistanceToCameraSD``.
   * - ``DistanceToCameraSD``, ``DistanceToImagePlaneSD``
     - EXR/NPY plus colorized PNG
     - Metric distance outputs.
   * - ``DiffuseAlbedoSD``
     - PNG/JPEG
     - Working RT2 alternative to ``DiffuseAlbedo``.
   * - ``Camera3dPositionSD``
     - EXR
     - Camera-space position.
   * - ``TargetMotionSD``
     - EXR
     - Working SDG alternative to ``Motion2d``.
   * - ``SemanticSegmentation``
     - NPY plus colorized PNG
     - Scalar semantic IDs.
   * - ``SemanticInstanceSegmentation``
     - NPY plus colorized PNG
     - 128-bit IDs as four uint32 channels.
   * - ``NonStableInstanceSegmentation``
     - NPY
     - Frame-local scalar instance IDs.
   * - ``StableIdSegmentation``
     - NPY plus colorized PNG
     - 128-bit stable IDs as four uint32 channels.

Additional Named Outputs
------------------------

``BasicWriter`` names the 98 additional ``sourceName`` values below. Each is
written to ``<frame>/<filename>`` when it is present and nonempty. A filename is
the ``sourceName`` converted to snake_case, except ``Depth`` and
``DiffuseAlbedo``, which keep the ``_native`` suffix to avoid colliding with
``DepthSD`` and ``DiffuseAlbedoSD``. Names that are not listed in this page are
written as the lowercased ``sourceName`` with a ``.npy`` extension.

PNG or JPEG (6 names)
^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - ``sourceName``
     - Filename
   * - ``DepthSensorPointCloudColor``
     - ``depth_sensor_point_cloud_color.png`` / ``depth_sensor_point_cloud_color.jpg``
   * - ``DiffuseAlbedo``
     - ``diffuse_albedo_native.png`` / ``diffuse_albedo_native.jpg``
   * - ``LdrColorSD``
     - ``ldr_color_sd.png`` / ``ldr_color_sd.jpg``
   * - ``Roughness``
     - ``roughness.png`` / ``roughness.jpg``
   * - ``SimpleShadingSD``
     - ``simple_shading_sd.png`` / ``simple_shading_sd.jpg``
   * - ``SpecularAlbedo``
     - ``specular_albedo.png`` / ``specular_albedo.jpg``

EXR (55 names)
^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - ``sourceName``
     - Filename
   * - ``BumpNormalResearch``
     - ``bump_normal_research.exr``
   * - ``Depth``
     - ``depth_native.exr``
   * - ``DepthSensorDistance``
     - ``depth_sensor_distance.exr``
   * - ``DepthSensorImager``
     - ``depth_sensor_imager.exr``
   * - ``DepthSensorPointCloudPosition``
     - ``depth_sensor_point_cloud_position.exr``
   * - ``DiffuseLightingDirectAndIndirect``
     - ``diffuse_lighting_direct_and_indirect.exr``
   * - ``DisocclusionMask``
     - ``disocclusion_mask.exr``
   * - ``DlssInput``
     - ``dlss_input.exr``
   * - ``EmissionAndForegroundMask``
     - ``emission_and_foreground_mask.exr``
   * - ``HighResDepth``
     - ``high_res_depth.exr``
   * - ``HighResDisocclusionMask``
     - ``high_res_disocclusion_mask.exr``
   * - ``HighResMotion2d``
     - ``high_res_motion2d.exr``
   * - ``HighResMotion2dXYZ``
     - ``high_res_motion2d_xyz.exr``
   * - ``HighResReflectionsMotion2d``
     - ``high_res_reflections_motion2d.exr``
   * - ``LightingComposited``
     - ``lighting_composited.exr``
   * - ``Motion2dDilated``
     - ``motion2d_dilated.exr``
   * - ``Motion2dXYZ``
     - ``motion2d_xyz.exr``
   * - ``Motion2dXYZDilated``
     - ``motion2d_xyz_dilated.exr``
   * - ``MotionWorld``
     - ``motion_world.exr``
   * - ``PtBackground``
     - ``pt_background.exr``
   * - ``PtDiffuseFilter``
     - ``pt_diffuse_filter.exr``
   * - ``PtDirectIllumation``
     - ``pt_direct_illumation.exr``
   * - ``PtGlobalIllumination``
     - ``pt_global_illumination.exr``
   * - ``PtIlluminance``
     - ``pt_illuminance.exr``
   * - ``PtLpe0``
     - ``pt_lpe0.exr``
   * - ``PtLpe1``
     - ``pt_lpe1.exr``
   * - ``PtLpe2``
     - ``pt_lpe2.exr``
   * - ``PtLpe3``
     - ``pt_lpe3.exr``
   * - ``PtLpe4``
     - ``pt_lpe4.exr``
   * - ``PtLpe5``
     - ``pt_lpe5.exr``
   * - ``PtLpe6``
     - ``pt_lpe6.exr``
   * - ``PtLpe7``
     - ``pt_lpe7.exr``
   * - ``PtLpe8``
     - ``pt_lpe8.exr``
   * - ``PtLuminance``
     - ``pt_luminance.exr``
   * - ``PtMotion``
     - ``pt_motion.exr``
   * - ``PtPreDenoisedResult``
     - ``pt_pre_denoised_result.exr``
   * - ``PtReflectionFilter``
     - ``pt_reflection_filter.exr``
   * - ``PtReflections``
     - ``pt_reflections.exr``
   * - ``PtRefractionFilter``
     - ``pt_refraction_filter.exr``
   * - ``PtRefractions``
     - ``pt_refractions.exr``
   * - ``PtSelfIllumination``
     - ``pt_self_illumination.exr``
   * - ``PtSubsurfaceFilter``
     - ``pt_subsurface_filter.exr``
   * - ``PtSubsurfaceScattering``
     - ``pt_subsurface_scattering.exr``
   * - ``PtViewNormal``
     - ``pt_view_normal.exr``
   * - ``PtVolumes``
     - ``pt_volumes.exr``
   * - ``PtWorldNormal``
     - ``pt_world_normal.exr``
   * - ``PtWorldPos``
     - ``pt_world_pos.exr``
   * - ``PtZDepth``
     - ``pt_z_depth.exr``
   * - ``ReflectionsFirstBounceDiffuseAlbedo``
     - ``reflections_first_bounce_diffuse_albedo.exr``
   * - ``ReflectionsFirstBounceNormal``
     - ``reflections_first_bounce_normal.exr``
   * - ``ReflectionsFirstBounceRayDirAndHitT``
     - ``reflections_first_bounce_ray_dir_and_hit_t.exr``
   * - ``ReprojectionQuality``
     - ``reprojection_quality.exr``
   * - ``SpecularLightingDirectAndIndirect``
     - ``specular_lighting_direct_and_indirect.exr``
   * - ``ViewspacePosition``
     - ``viewspace_position.exr``
   * - ``XRDepth``
     - ``xr_depth.exr``

NPY (37 names)
^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - ``sourceName``
     - Filename
   * - ``BoundingBox2DLooseSD``
     - ``bounding_box_2d_loose_sd.npy``
   * - ``BoundingBox2DTightSD``
     - ``bounding_box_2d_tight_sd.npy``
   * - ``BoundingBox3DSD``
     - ``bounding_box_3d_sd.npy``
   * - ``Caustics``
     - ``caustics.npy``
   * - ``CrossCorrespondenceSD``
     - ``cross_correspondence_sd.npy``
   * - ``DeveloperDebugTexture``
     - ``developer_debug_texture.npy``
   * - ``DirectDiffuse``
     - ``direct_diffuse.npy``
   * - ``DirectSpecular``
     - ``direct_specular.npy``
   * - ``DlssInputWithoutTransparency``
     - ``dlss_input_without_transparency.npy``
   * - ``Exposure``
     - ``exposure.npy``
   * - ``GBufferDepth``
     - ``g_buffer_depth.npy``
   * - ``IndirectDiffuse``
     - ``indirect_diffuse.npy``
   * - ``IndirectDiffuseFirstBounceNormal``
     - ``indirect_diffuse_first_bounce_normal.npy``
   * - ``IndirectDiffuseFirstBounceRayDirAndHitT``
     - ``indirect_diffuse_first_bounce_ray_dir_and_hit_t.npy``
   * - ``InstanceIdTokenMapSD``
     - ``instance_id_token_map_sd.npy``
   * - ``Irradiance``
     - ``irradiance.npy``
   * - ``OcclusionSD``
     - ``occlusion_sd.npy``
   * - ``PtMultiMatte0``
     - ``pt_multi_matte0.npy``
   * - ``PtMultiMatte1``
     - ``pt_multi_matte1.npy``
   * - ``PtMultiMatte2``
     - ``pt_multi_matte2.npy``
   * - ``PtMultiMatte3``
     - ``pt_multi_matte3.npy``
   * - ``PtMultiMatte4``
     - ``pt_multi_matte4.npy``
   * - ``PtMultiMatte5``
     - ``pt_multi_matte5.npy``
   * - ``PtMultiMatte6``
     - ``pt_multi_matte6.npy``
   * - ``PtMultiMatte7``
     - ``pt_multi_matte7.npy``
   * - ``Reflections``
     - ``reflections.npy``
   * - ``ReflectionsFirstBounceWorldPos``
     - ``reflections_first_bounce_world_pos.npy``
   * - ``ReflectionsHitT``
     - ``reflections_hit_t.npy``
   * - ``ReflectionsMotion2d``
     - ``reflections_motion2d.npy``
   * - ``ReflectionsMotionWorld``
     - ``reflections_motion_world.npy``
   * - ``SemanticIdMapDeltas``
     - ``semantic_id_map_deltas.npy``
   * - ``StableDepth``
     - ``stable_depth.npy``
   * - ``StableId``
     - ``stable_id.npy``
   * - ``StableIdMapDeltas``
     - ``stable_id_map_deltas.npy``
   * - ``SubsurfaceScattering``
     - ``subsurface_scattering.npy``
   * - ``TruncationSD``
     - ``truncation_sd.npy``
   * - ``ViewMatricesAux``
     - ``view_matrices_aux.npy``

Decoded Buffers
---------------

The adapter decodes these buffers when ``decode=True``. ``BasicWriter`` writes
the decoded representation as JSON.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - ``sourceName``
     - Decoded representation
   * - ``SemanticBoundingBox2DTight``, ``SemanticBoundingBox2DLoose``
     - Structured records with 128-bit instance/semantic IDs and pixel bounds.
   * - ``SemanticBoundingBox3D``
     - Structured records with IDs, 3-D bounds, and a 4x4 transform.
   * - ``SemanticIdMap``
     - ``dict[int, str]`` mapping semantic ID to label.
   * - ``StableIdMap``
     - ``dict[int, str]`` mapping stable ID to USD prim path.
   * - ``StableIdSemanticIdMap``
     - Structured 32-byte stable-ID/semantic-ID pair records.
   * - ``InstanceMap``
     - Structured instance index to stable-semantic-table index records; ``0xffffffff`` is invalid.
   * - ``CameraParams``
     - Structured camera matrices, resolution, clipping, and lens fields.

Static-Scene Delta Buffers
--------------------------

``SemanticIdMapDeltas`` and ``StableIdMapDeltas`` are mapped as four-byte zero-count
payloads after static warm-up. They are not decoded automatically because a
non-empty delta layout has not yet been runtime-verified.

Other Observed Outputs
----------------------

The ovrtx 0.5 robot capture returned image-shaped data for ``DiffuseAlbedo``,
``SpecularAlbedo``, ``Roughness``, ``DepthLinearized``,
``EmissionAndForegroundMask``, lighting decomposition, and many ``Pt*`` passes.
The ``PtMultiMatte0``–``PtMultiMatte7`` buffers were zero-size in that frame.
``DirectDiffuse``, ``DirectSpecular``, ``IndirectDiffuse``, and several other
legacy passes were also zero-size. ``BasicWriter`` can name these outputs if
they become nonempty, but does not synthesize data for them.

The nonempty 1-D ``BoundingBox*SD``, ``OcclusionSD``, ``TruncationSD``,
``InstanceIdTokenMapSD``, and map-delta buffers remain ``.npy``. Their shapes
do not make them empty; their binary record formats require separate decoding
before they can be represented usefully as JSON or an image.

Sensor-Only Outputs
-------------------

``PointCloud``, ``RtxSensorCpu``, and ``RtxSensorGpu`` are not camera outputs.
Route them through the corresponding lidar/radar workflow instead.

``PointCloud`` is a composite output: ``OvrtxOutput`` returns it as a
``{"tensors": ..., "params": ...}`` mapping of named channels
(``Coordinates``, ``Intensity``, ``Counts``, ...), and ``BasicWriter`` stores it
as ``point_cloud.npz`` with one archive member per tensor and a ``param_``
prefix on each param.

ovrtx 0.5 removed the ``GenericModelOutput`` render var; a render product that
requests it is skipped with a diagnostic. Author ``PointCloud`` instead.
