from __future__ import annotations

import json
from typing import Any, Dict, Optional, Union, Sequence, Tuple
import numpy as np
from pathlib import Path

from scipy.spatial.transform import Rotation as R
from numpy.linalg import inv, norm, det
import cv2


import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    # pip install planetary-data-reader
    from pdr import read as pdr_read
except ImportError as e:
    raise ImportError(
        "Planetary Data Reader (pdr) is required. Install via: pip install planetary-data-reader") from e

try:
    # pip install colour_demosaicing
    import colour_demosaicing
except ImportError as e:
    raise ImportError(
        "Colour colour_demosaicing is required. Install via: pip install colour_demosaicing") from e


# from readers import find_file_from_parent, interpolate_csv_xy, find_waypoint_for_site_drive
from readers import *

class MPPP_Image:
    """
    Container for one PDS IMG and its key metadata/state, read via PDR.

    Design:
      - PDR-backed, supports PDS3/PDS4.
      - Non-destructive by default (DNs preserved). Radiometry is opt-in.
      - No inference for camera/sol/site/drive — you must provide these.

    You may provide identifiers via constructor args or config:
      config = {
        "identifiers": {"camera": "NL", "sol": 1623, "site": 53, "drive": 123},
        "radiometry": {"apply_dn_to_radiance": false, "output_dtype": "float64"}
      }
    """

    IMG_path: Path
    filename: str
    stem: str
    label: Dict[str, Any]
    dn: Optional[np.ndarray]
    image: Optional[np.ndarray]
    dtype_original: Optional[str]

    # identifiers (must be provided; no inference)
    cam: Optional[str]
    sol: Optional[int]
    site: Optional[int]
    drive: Optional[int]

    LMST: Optional[str]
    L_s: Optional[float]  # solar longitude (degrees), if available

    scale_to_radiance: Optional[float]
    radiometry_params: Dict[str, Any]

    frame: str
    config: Dict[str, Any]

    def __init__(
        self,
        IMG_path: Union[str, Path],
        config: Optional[Union[str, Path, Dict[str, Any]]] = None,
        waypoints: Optional[Dict[str, Any]] = None,
        *,
        camera: Optional[str] = None,
        sol: Optional[int] = None,
        site: Optional[int] = None,
        drive: Optional[int] = None,
        just_label: bool = False,
        frame: str = "site3",
    ) -> None:
        
        self.IMG_path = Path(IMG_path)
        self.filename_path = Path(IMG_path)

        self.filename = self.IMG_path.name
        self.stem = self.IMG_path.stem
        self.frame = frame
        self.config = self._load_config(config)
        del config


        # Read via PDR (dict-like: keys commonly include 'LABEL' and 'IMAGE')
        data = pdr_read(str(self.IMG_path))

        # Normalize label to a plain dict
        self.label = self._coerce_label_to_dict(self.IMG_path, data["LABEL"])

        # Identifiers: no inference. Pull from args, else config.identifiers
        self.cam   = self.filename[:3] + self.filename[45:48]
        self.sol   = int(self.filename[4:8])
        self.site  = int( self.label['ROVER_MOTION_COUNTER'][0] )
        self.drive = int( self.label['ROVER_MOTION_COUNTER'][1] )

        self.down_sample = int(self.filename.split('_')[-1][3])

        waypoint = find_waypoint_for_site_drive(waypoints, self.site, self.drive )
        try: self.waypoint = waypoint['properties']
        except: self.waypoint = waypoint

        self.waypoint_site3 = waypoints['features'][0]['properties']

        if self.waypoint['drive'] == self.drive:

            self.relative_northing  = self.waypoint["northing"]   - self.waypoint_site3["northing"] 
            self.relative_easting   = self.waypoint["easting"]    - self.waypoint_site3["easting"] 
            self.relative_elevation = self.waypoint["elev_geoid"] - self.waypoint_site3["elev_geoid"] 

        else:
            self.waypoint = find_waypoint_for_site_drive(waypoints, self.site, 0)['properties']
            self.relative_northing  = self.waypoint["northing"]   - self.waypoint_site3["northing"]   + self.label['ROVER_COORDINATE_SYSTEM']['ORIGIN_OFFSET_VECTOR'][0]
            self.relative_easting   = self.waypoint["easting"]    - self.waypoint_site3["easting"]    + self.label['ROVER_COORDINATE_SYSTEM']['ORIGIN_OFFSET_VECTOR'][1]
            self.relative_elevation = self.waypoint["elev_geoid"] - self.waypoint_site3["elev_geoid"] - self.label['ROVER_COORDINATE_SYSTEM']['ORIGIN_OFFSET_VECTOR'][2]


        # set camera model
        self.cmod_from_cahvor_label()
        try:
            self.cmod_for_landing_frame()
        except:
            print('skipping cmods')

        # LMST if present in label; optional L_s (solar longitude) for provenance
        self.LMST = self._get_label_scalar(self.label, ["LOCAL_MEAN_SOLAR_TIME.LMST"])
        self.L_s = np.float64( self._get_label_scalar(self.label, ["SOLAR_LONGITUDE"]))

        # self.solar_az = self.label['SITE_DERIVED_GEOMETRY_PARMS']['SOLAR_AZIMUTH'][0]
        # self.solar_el = self.label['SITE_DERIVED_GEOMETRY_PARMS']['SOLAR_ELEVATION'][0]        
        self.solar_az = np.float64( self._get_label_scalar(self.label, ['SITE_DERIVED_GEOMETRY_PARMS.SOLAR_AZIMUTH']) )
        self.solar_el = np.float64( self._get_label_scalar(self.label, ['SITE_DERIVED_GEOMETRY_PARMS.SOLAR_ELEVATION']) )

        # Estimated tau values 
        path_csv_taus_versus_L_s = find_file_from_parent('params',"M2020_taus_versus_L_s.csv",)
        self.tau_estimated = interpolate_csv_xy( path_csv_taus_versus_L_s, x_query = self.L_s)

        try:
            self.tau_ref = self.config["radiometry"]["tau_reference"]
        except: self.tau_ref = 0.3

        self.solar_mu = np.sin(np.deg2rad(self.solar_el))
        self.scale_zenith_estimated = self.solar_mu * np.exp( -(self.tau_estimated-self.tau_ref)/6/self.solar_mu )

        # self.scale_int16_to_rad = self.label["DERIVED_IMAGE_PARMS"]["RADIANCE_SCALING_FACTOR"][0]
        self.scale_int16_to_rad  = np.float64( self._get_label_scalar(self.label, ["DERIVED_IMAGE_PARMS.RADIANCE_SCALING_FACTOR","RADIANCE_SCALING_FACTOR"] ))
        self.offset_int16_to_rad = np.float64( self._get_label_scalar(self.label, ["DERIVED_IMAGE_PARMS.RADIANCE_OFFSET","RADIANCE_OFFSET"] ))

        # Image versions
        self.dtype_original = None      # the origninal data type expected to be uint 16
        self.image_original = None      # the origninal 16 bit image from the PDS
        self.image_rad      = None      # radiance with color corection in 32 bit float 
        self.image_int16    = None      # image_flt32 scaled to 16 bit uint with scale_rad_to_int16
        self.image_int8     = None      # image_int16 offset, scaled, and gamma corrected to 8 bit uint using scale_int16_to_int8, offset_int16_to_int8, gamma_int16_to_int8
        self.mask_valid     = None      # valid pixel mask in uint8. invalid: 0 and valid:255
        self.mask           = None      # reconstruction mask. exclude: 0 and include:255

        # if not just_label:

        arr = np.asarray(data["IMAGE"])
        arr = self._ensure_hw_or_hwc(arr)
        self.dtype_original = str(arr.dtype)
        self.image_original = arr

        # convert image to radiance
        # print( self.scale_int16_to_rad, self.offset_int16_to_rad)
        self.image_rad = np.float64(self.image_original) * self.scale_int16_to_rad + self.offset_int16_to_rad


        # ensure that the image has three bands
        if len(self.image_original.shape) == 2:
            arr = self.image_rad
            if self.cam[0] in ['N','F']:
                self.image_rad = np.stack( [arr,arr,arr], axis=-1)
            if self.cam[0] in ['Z','S']:
                self.image_rad = colour_demosaicing.demosaicing_CFA_Bayer_Malvar2004( arr, 'RGGB' )

        # create valid pixel mask
        self.mask_valid = np.zeros( self.image_original.shape, dtype=np.uint8 )
        self.mask_valid[ self.image_original != 0 ] =  255
        if len(self.mask_valid.shape) == 3:
            self.mask_valid = np.prod(self.mask_valid, axis=2)
        self.mask_valid = np.uint8( self.mask_valid )
        self.mask = ( self.mask_valid ).copy()

        # apply estimated zenith scaling
        if self.config["radiometry"]["apply_tau_correction"]:
            self.image_rad /= self.scale_zenith_estimated

        # global color scale
        self.scale_rad_to_int16   = np.float64(self.config["color"]["scale_rad_to_int16"])
        self.scale_int8_to_int16  = np.float64(self.config["color"]["scale_int8_to_int16"])
        self.offset_int8_to_int16 = np.float64(self.config["color"]["offset_int8_to_int16"])

        # white balance (channel gains) 
        if self.config["color"]["enhance"]:
            # print(self.cam[:3])
            if self.cam[2] == "M":
                r, g, b = self.config["color"]["white_balance_vce"]
            elif self.cam[0] in ["N", "F", "R"]:
                r, g, b = self.config["color"]["white_balance_ecam"]
            elif self.cam[0] in ["Z"]:
                r, g, b = self.config["color"]["white_balance_zcam"]
            else:
                r, g, b = [ 1.0, 1.0, 1.0 ]
        else:
            r, g, b = [ 1.0, 1.0, 1.0 ]

        self.color_balance = np.array([r, g, b], dtype=np.float64).reshape(1, 1, 3)
        self.image_rad  *= self.color_balance
        self.image_rad   = self._zero_invalid_pixels( self.image_rad, self.mask_valid, 1.1*(self.scale_int8_to_int16/self.scale_rad_to_int16) ) 


        # Infer mask
        if self.config["masking"]["infer_mask"]:

            # image on which to run inferance. Should betwee float32 with apprximately athe same intereger scales as an uint8 image. 
            # It gets normalized, so it is alright if some values are below zeros or above 255.
            image_for_predict = np.array( self.image_rad * self.scale_rad_to_int16 / self.scale_int8_to_int16 + self.offset_int8_to_int16 ) 
            image_for_predict = np.clip( image_for_predict, 0, 255)
            # self.image_for_predict  = image_for_predict


            # checkpoint_template = find_file_from_parent('masks/m20/sam2',self.config["masking"]["checkpoint_template"])
            # checkpoint_trained  = find_file_from_parent('masks/m20/sam2',self.config["masking"]["checkpoint_trained"])
            # checkpoint_config   = find_file_from_parent('masks/m20/sam2',self.config["masking"]["checkpoint_config"])
            # predictor = build_mask_predictor( checkpoint_template, checkpoint_trained, checkpoint_config )
            # mask_pred, image_from_predict = predict_mask_with_auto_points( predictor, image_for_predict, True )


            checkpoint_trained  = find_file_from_parent('masks/m20',self.config["masking"]["checkpoint_trained"])
            # print( checkpoint_trained)


            # Load trained model weights
            device = "cpu"
            fpn_width = 256
            target_size= 1648
            threshold = 0.4
            # model = ConvNeXtSeg( pretrained=False, fpn_width=fpn_width)
            # state = torch.load( checkpoint_trained, map_location=device)
            # model.load_state_dict(state["model"], strict=True)
            mask_pred, prob_pred = infer_segmentation(model, image_for_predict, device="cpu", target_size=target_size, threshold=threshold )
            mask_pred = dilate_probability_map( prob_pred, threshold = threshold, kernel_size = 3 )

            self.mask = mask_pred.copy()
            self.mask[ self.mask_valid == 0 ] = 0

            if self.mask.shape[1] == 1648:
                self.mask[0:5,0] = 0
                self.mask[-1:,0]  = 0
            

            # # refined mask with SAM
            # self.mask = np.zeros( mask_pred.shape ).astype("uint8")
            # self.mask[ mask_pred > 0 ] = 255
            # self.mask[ self.mask_valid == 0 ] = 0


        # pad image to standard dimensions

        self.im_shape_orig = self.image_rad.shape
        self.full_height, self.full_width = [self.h,self.w]

        self.down_sample_scale  = 1
        if self.down_sample==1: self.down_sample_scale = 1/2
        if self.down_sample==2: self.down_sample_scale = 1/4

        if self.config["resize"]["apply_padding"]:
            if self.cam[0] in ["N", "F", "R"]:
                

                # print( self.down_sample, self.down_sample_scale)
                
                self.full_height, self.full_width = [ int(3840*self.down_sample_scale), int(5120*self.down_sample_scale) ]

                # print( 'target image size',  self.full_height, self.full_width, "down_sample", self.down_sample, self.down_sample_scale,)
                try:
                    self.pad_left   = int( self.label['INSTRUMENT_STATE_PARMS']['TILE_FIRST_LINE_SAMPLE'] - 1 )
                except:
                    self.pad_left   = int( np.min(self.label['INSTRUMENT_STATE_PARMS']['TILE_FIRST_LINE_SAMPLE']) - 1 )
                try:
                    self.pad_top    = int( self.label['INSTRUMENT_STATE_PARMS']['TILE_FIRST_LINE'] - 1 )  
                except:
                    self.pad_top    = int( np.min(self.label['INSTRUMENT_STATE_PARMS']['TILE_FIRST_LINE']) - 1 )  


                if (self.pad_top+self.im_shape_orig[0]) > self.full_height:
                    print( "Error: (self.pad_top+self.im_shape_orig[0]) > self.full_height ")
                    self.pad_top = self.full_height - self.im_shape_orig[0]

                # special case for NLF_0474_0709027186_885RAD_N0260850NCAM13474_0A0195J01, don't know why it works
                if self.pad_left >= self.full_width:
                    print( "Warning, self.pad_left >= self.full_width")
                    self.pad_left = int(self.pad_left/2)

                self.pad_right  = int( self.full_width  - self.pad_left -  self.im_shape_orig[1] )
                self.pad_bottom = int( self.full_height - self.pad_top  -  self.im_shape_orig[0] )
                
                if self.pad_right < 0: self.pad_right = 0
                if self.pad_bottom < 0: self.pad_bottom = 0

            if self.cam[0] in ["Z", "S"]: 

                self.pad_left, self.pad_right, self.pad_top, self.pad_bottom = [ 0.0, 0.0, 0.0, 0.0 ]

                self.full_height, self.full_width = [ 1200, 1648 ]
   
                self.pad_left   =                    self.label['MINI_HEADER']['FIRST_LINE_SAMPLE'] - 1
                self.pad_right  = self.full_width  - self.label['MINI_HEADER']['LINE_SAMPLES']      - self.label['MINI_HEADER']['FIRST_LINE_SAMPLE']  + 1
                self.pad_top    =                    self.label['MINI_HEADER']['FIRST_LINE']        - 1
                self.pad_bottom = self.full_height - self.label['MINI_HEADER']['LINES']             - self.label['MINI_HEADER']['FIRST_LINE']         + 1

                # self.full_height_new = int( self.full_width / self.config["resize"]["standard_ratio"])
            #     self.pad_bottom += int( (self.full_height_new - self.full_height)/2 )
            #     self.pad_top    += int( (self.full_height_new - self.full_height)/2 )

            #     self.full_height = self.full_height_new

            self.im_shape_new   = ( self.full_height, self.full_width, self.im_shape_orig[2] )

            if self.pad_top!=0 or self.pad_bottom!=0 or self.pad_left!=0 or self.pad_right!=0:

                
                self.image_rad  = pad_image(  self.image_rad.copy(),  [self.pad_left, self.pad_right, self.pad_top, self.pad_bottom ] )
                self.mask_valid = pad_image(  self.mask_valid.copy(), [self.pad_left, self.pad_right, self.pad_top, self.pad_bottom ] )
                self.mask       = pad_image(  self.mask      .copy(), [self.pad_left, self.pad_right, self.pad_top, self.pad_bottom ] )

                self.h    = self.image_rad.shape[0]
                self.w    = self.image_rad.shape[1]
                self.hc  += self.pad_left
                self.cx  += self.pad_left
                self.cxp += self.pad_left
                self.vc  += self.pad_top
                self.cy  += self.pad_top
                self.cyp += self.pad_top
                self.K_cam[0,2] += self.pad_left
                self.K_cam[1,2] += self.pad_top

                print( 'resizing image {} to {} by padding = [ left, right, top, bottom ] = [ {}, {}, {}, {} ]'.format( \
                    self.im_shape_orig, self.im_shape_new, self.pad_left, self.pad_right, self.pad_top, self.pad_bottom ))
                
        
        
        self.h    = self.image_rad.shape[0]
        self.w    = self.image_rad.shape[1]



        # print( self.image_rad.shape)

        if self.config["camera_model"]["intrinsics_from_xml"]:

            if self.cam[0] not in ['N']:
                print( f"Skipping loading intrinsics from XML file. {self.cam[0]} is not yet supported.") 
                # pass

            else:
                path_cmod_xml = find_file_from_parent('params/m20_cmods',"M2020_N"+self.cam[1]+"1_frame.xml")
                print( self.cam[0],"replacing intrinsic parameters with those in", path_cmod_xml)

                # self.cmod_intrinsics_from_xml(path_cmod_xml)

                cmod_xml = read_xml(path_cmod_xml)

                for key in ("k1", "k2", "k3", "k4","p1", "p2", "b1", "b2", "cx", "cy"):
                    cmod_xml.setdefault(key, 0.0)

                scale  = 2*self.down_sample_scale

                cmod_xml['width']  *= scale
                cmod_xml['height'] *= scale
                cmod_xml['f']      *= scale
                cmod_xml['cx']     *= scale
                cmod_xml['cy']     *= scale
                cmod_xml['b1']     *= scale
                cmod_xml['b2']     *= scale

                if self.h != cmod_xml['height'] or self.w != cmod_xml['width']:
                    raise ValueError( f"width or height mismatch. Got ({int(cmod_xml['height'])},{int(cmod_xml['width'])}) but expeted ({self.h},{self.w}). Relative scale {scale}. Down sample {self.down_sample}.")

                self.k1 = cmod_xml['k1']
                self.k2 = cmod_xml['k2']
                self.k3 = cmod_xml['k3']
                self.k4 = cmod_xml['k4']
                self.k5 = 0.0
                self.k6 = 0.0

                self.p1 = cmod_xml['p1']
                self.p2 = cmod_xml['p2']      

                self.d_cam = [ self.k1, self.k2, self.p1, self.p2, self.k3, self.k4, self.k5, self.k6 ]
                # self.d_cam = np.float64(self.d_cam)

                self.cxp = cmod_xml['cx']
                self.cyp = cmod_xml['cy']
                
                self.cx = cmod_xml['cx'] + cmod_xml['width']/2
                self.cy = cmod_xml['cy'] + cmod_xml['height']/2

                self.f   = cmod_xml['f']
                self.b1  = cmod_xml['b1']
                self.b2  = cmod_xml['b2']
                self.fx  = cmod_xml['f'] + cmod_xml['b1']
                self.fy  = cmod_xml['f']

                self.K_cam = np.array([
                        [ self.fx, self.b2, self.cx ],
                        [ 0, self.fy, self.cy ],
                        [ 0,      0,      1 ], ])
    

        if self.config["resize"]["extra_padding"] and self.config["resize"]["apply_padding"]:

            self.pad_left   = int(self.w*self.config["resize"]["extra_fraction"]/2)
            self.pad_right  = self.pad_left
            self.pad_top    = int(self.h*self.config["resize"]["extra_fraction"]/2)
            self.pad_bottom = self.pad_top


            self.image_rad = np.float32( self.image_rad )
            self.im_shape_pre_extra_padding = self.image_rad.shape

            # print( self.mask.shape)
            
            self.image_rad  = pad_image(  self.image_rad .copy(), [self.pad_left, self.pad_right, self.pad_top, self.pad_bottom ] )
            self.mask_valid = pad_image(  self.mask_valid.copy(), [self.pad_left, self.pad_right, self.pad_top, self.pad_bottom ] )
            self.mask       = pad_image(  self.mask      .copy(), [self.pad_left, self.pad_right, self.pad_top, self.pad_bottom ] )

            self.h    = self.image_rad.shape[0]
            self.w    = self.image_rad.shape[1]
            self.hc  += self.pad_left
            self.cx  += self.pad_left
            self.cxp += self.pad_left
            self.vc  += self.pad_top
            self.cy  += self.pad_top
            self.cyp += self.pad_top
            self.K_cam[0,2] += self.pad_left
            self.K_cam[1,2] += self.pad_top

            self.full_height, self.full_width = [ self.h, self.w  ]
            self.im_shape_new  = self.image_rad.shape

            print( 'resizing image {} to {} by padding = [ left, right, top, bottom ] = [ {}, {}, {}, {} ]'.format( \
                self.im_shape_pre_extra_padding, self.im_shape_new, self.pad_left, self.pad_right, self.pad_top, self.pad_bottom ))
            
                


        # Undistort
        self.undistorted = self.config["resize"]["undistort"]
        self.recenter_principal_point = self.config["resize"]["recenter_principal_point"]


        if self.undistorted:

            # print( "K_cam", self.K_cam)

            center_projection = self.config["resize"]["recenter_principal_point"]

            self.image_rad, K_new  = undistort_isotropic( 
                image=self.image_rad.copy(),  K=self.K_cam.copy(), dist_coeffs=self.d_cam.copy(), f_strategy="mean", center_projection=self.recenter_principal_point, interpolation=cv2.INTER_LINEAR, )
            self.mask_valid, K_new  = undistort_isotropic( 
                image=self.mask_valid.copy(),  K=self.K_cam.copy(), dist_coeffs=self.d_cam.copy(), f_strategy="mean", center_projection=self.recenter_principal_point, interpolation=cv2.INTER_LINEAR)
            self.mask     , K_new  = undistort_isotropic( 
                image=self.mask      .copy(),  K=self.K_cam.copy(), dist_coeffs=self.d_cam.copy(), f_strategy="mean", center_projection=self.recenter_principal_point, interpolation=cv2.INTER_LINEAR)
            
            self.mask_valid[ self.mask_valid < 255 ] = 0
            self.mask      [ self.mask       < 255 ] = 0

            # update intrinsic camera parameters after undistortion

            self.K_cam = K_new.copy()
            self.k1, self.k2, self.p1, self.p2, self.k3, self.k4, self.k5, self.k6 = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
            self.d_cam = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]

            self.cx  = K_new[0,2]
            self.cy  = K_new[1,2]
            self.cxp = K_new[0,2] - self.w/2
            self.cyp = K_new[1,2] - self.h/2

            self.f   = K_new[1,1]
            self.b1  = K_new[0,0] - K_new[1,1]
            self.b2  = K_new[0,1]
            self.fx  = K_new[0,0]
            self.fy  = K_new[1,1]


        # convert rad float to int16 linear
        self.image_int16 = np.clip( self.image_rad * self.scale_rad_to_int16, 0, 256**2-1 ).astype(np.uint16)
        self.image_int16 = self._zero_invalid_pixels( self.image_int16, self.mask_valid, 256 )

        # convert int16 linear to int8 with gamma correction
        arr = self.image_rad * self.scale_rad_to_int16 / self.scale_int8_to_int16 + self.offset_int8_to_int16
        if self.config["color"]["apply_gamma"]:
            arr = ( arr )**(1/self.config["color"]["gamma"]) * (256)**(1-1/self.config["color"]["gamma"])
        
        self.image_int8 = np.clip( arr, 0, 255 ).astype(np.uint8)
        self.image_int8  = self._zero_invalid_pixels( self.image_int8, self.mask_valid, 1 )

        # RGBA mask is the predicted mask
        if True:
            self.image_int16_rgba = stack_image_with_mask( self.image_int16, np.uint16(self.mask))
            self.image_int8_rgba  = stack_image_with_mask( self.image_int8,  self.mask )

        
        try:
            self.reference = [ self.filename[:-4]+".png" ] + list( self.C_ref_site ) + list( self.angles_ypr )
            print( "Reference XYZYPR:", "\t".join(map(str,  self.reference)) )
        except:
            self.reference = None
            print( "No references")
            


        

        # output camera model params
        if False:
            
            print( "R_ref", self.R_ref_site)
            print( "t_ref", self.C_ref_site)
            print( "K_cam", self.K_cam)
            print( "d_cam", self.d_cam)
            # print( )
            print( "Reference XYZYPR:", "\t".join(map(str,  self.reference)) )
            # print( )
            # print( "Reference position XYZ=", self.C_ref_site )
            # print( "Reference angles   YPR=", self.angles_ypr )
            # print( )






    # -------------------------
    # Helper functions
    # -------------------------

    # def cmod_from_xml_file( self, xml_file ):


    def cmod_from_cahvor_label( self ):
                
        label_cmod = self.label["GEOMETRIC_CAMERA_MODEL"]
        self.C  = np.array(  label_cmod['MODEL_COMPONENT_1'], dtype=np.float64 )
        self.A  = np.array(  label_cmod['MODEL_COMPONENT_2'], dtype=np.float64 )
        self.H  = np.array(  label_cmod['MODEL_COMPONENT_3'], dtype=np.float64 )
        self.V  = np.array(  label_cmod['MODEL_COMPONENT_4'], dtype=np.float64 )
        self.O  = np.array(  label_cmod['MODEL_COMPONENT_5'], dtype=np.float64 )
        self.R  = np.array(  label_cmod['MODEL_COMPONENT_6'], dtype=np.float64 )

        self.hs = norm( np.cross( self.H, self.A ) )
        self.vs = norm( np.cross( self.V, self.A ) )
        self.hc = np.dot( self.H, self.A ) 
        self.vc = np.dot( self.V, self.A ) 

        self.hp = ( self.H - self.hc* self.A ) / self.hs
        self.vp = ( self.V - self.vc* self.A ) / self.vs

        self.theta = np.arcsin( ( - norm( np.cross( self.vp, self.hp ) )
                                  / norm( self.vp )
                                  / norm( self.hp ) ) )
        self.theta_degrees = np.rad2deg( self.theta )
        
        self.K_cam = np.array([
                    [ -self.hs*np.sin(self.theta), self.hs*np.cos(self.theta), self.hc ],
                    [                           0,                    self.vs, self.vc ],
                    [                           0,                          0,       1 ], ])

        self.rot_cam = np.matmul( inv( self.K_cam ), 
                                  np.array( [ self.H, self.V, self.A ] ) )
        
        R_cam2ned = np.array( [[0,-1,0],[1,0,0],[0,0,1]] )
        # self.R_ref = R_cam2ned @ self.rot_cam 
        # self.t_ref = R_cam2ned @ self.C
        
        self.R_cam = R.from_matrix( self.rot_cam )       
        self.R_veh2cam = self.R_cam
        
        self.R_ned2enu = R.from_matrix( [[0,1,0],[1,0,0],[0,0,-1]] )
        
        self.w, self.h = [ self.label['IMAGE']['LINE_SAMPLES'], self.label['IMAGE']['LINES'] ]
        # if self.w==1600: self.w = 1648

        self.k1 = self.R[1]
        self.k2 = self.R[2]
        self.k3 = 0
        self.k4 = 0
        self.k5 = 0
        self.k6 = 0

        self.p1 = 0
        self.p2 = 0        

        self.d_cam = [ self.k1, self.k2, self.p1, self.p2, self.k3, self.k4, self.k5, self.k6 ]

        self.cx = self.hc 
        self.cy = self.vc 
        
        self.cxp = self.hc - self.w/2
        self.cyp = self.vc - self.h/2
        
        self.f  =  self.vs
        self.b1 = -self.hs * np.sin( self.theta ) - self.vs
        self.b2 =  self.hs * np.cos( self.theta )

        self.fx  = self.f + self.b1
        self.fy  = self.f

        
    def cmod_for_landing_frame( self ):

        # initiate rotation R_ref_site and location C_ref_site in the landing frame (a.k.a site=3 drive=0)

        # if self.frame == 'site3':
            
        self.az     =   self.label['SITE_DERIVED_GEOMETRY_PARMS' ]['INSTRUMENT_AZIMUTH'][0]
        self.az_veh = ( self.label['ROVER_DERIVED_GEOMETRY_PARMS']['INSTRUMENT_AZIMUTH'][0] - 
                        self.label['SITE_DERIVED_GEOMETRY_PARMS' ]['INSTRUMENT_AZIMUTH'][0])%360

        self.q_site  = q_wxyz2xyzw( self.label['ROVER_COORDINATE_SYSTEM']['ORIGIN_ROTATION_QUATERNION'] )
        self.R_rnav2site = R.from_quat( self.q_site )

        # if self.waypoint is not None:
        #     self.relative_northing  = self.waypoint["northing"] - self.waypoint_site3["northing"] 
        #     self.relative_easting   = self.waypoint["easting"] - self.waypoint_site3["easting"] 
        #     self.relative_elevation = self.waypoint["elev_geoid"] - self.waypoint_site3["elev_geoid"] 
        # else: 
        #     self.relative_northing  = 0.0
        #     self.relative_easting   = 0.0
        #     self.relative_elevation = 0.0

        self.R_cam_site = (self.R_rnav2site.apply( self.rot_cam, inverse=0 ) )
        self.C_cam_site = self.R_rnav2site.apply( self.C, inverse=0 ) \
                        + np.array( [self.relative_northing, self.relative_easting , -self.relative_elevation ])

        R_cam2ned = np.array( [[0,-1,0],[1,0,0],[0,0,1]] )
        R_ned2ned = np.array( [[0,1,0],[1,0,0],[0,0,-1]] )
        self.R_ref_site = R_ned2ned @ self.R_cam_site
        self.C_ref_site = R_ned2ned @ self.C_cam_site #+ np.array( [ self.relative_easting , self.relative_northing, self.relative_elevation ])
            
        # if self.frame == 'rnav':

        self.angles_ypr = find_ypr_from_R_ref( self.R_ref_site )
        if self.angles_ypr[0] < 0.0: 
            self.angles_ypr[0] = self.angles_ypr[0] + 360
        # self.angles_opk = find_opk_from_R_ref( self.R_ref_site )
        

    def _zero_invalid_pixels( self, arr: np.ndarray, mask: np.narray, not_zero: float ) -> np.ndarray:
          mask3 = np.stack([ mask, mask, mask], axis=2 )
          arr[ (mask3 == False) ] = 0
          arr[ (mask3 == True)*(arr <= 0) ] = not_zero
          return arr


    def _load_config(self, config: Optional[Union[str, Path, Dict[str, Any]]]) -> Dict[str, Any]:
        # if config is None:
        #     return {}
        if isinstance(config, dict):
            return config
        p = Path(config)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("Config JSON must be a top-level object.")
        return data

    @staticmethod
    def _coerce_label_to_dict(path: Path, label_raw: Any) -> Dict[str, Any]:
        """
        Accept PVL/XML text, bytes, or dict-like; return a plain Python dict.
        Falls back to sibling .XML/.LBL files if needed.
        """
        # Already a dict
        if isinstance(label_raw, dict):
            return label_raw

        # xarray-like with attrs
        attrs = getattr(label_raw, "attrs", None)
        if isinstance(attrs, dict):
            # common PDR patterns: attrs['label'], attrs['LABEL'], or label embedded directly
            for k in ("label", "LABEL"):
                if k in attrs and isinstance(attrs[k], (dict, str, bytes)):
                    return MPPP_Image._coerce_label_to_dict(path, attrs[k])
            # as last resort, attrs itself
            return dict(attrs)

        # bytes → str
        if isinstance(label_raw, (bytes, bytearray)):
            text = label_raw.decode("utf-8", errors="ignore")
        else:
            text = str(label_raw)

        text_stripped = text.lstrip()

        # Try XML (PDS4)
        if text_stripped.startswith("<"):
            try:
                import xmltodict  # lazy import
            except ImportError as e:
                raise ImportError("xmltodict is required for PDS4 label parsing (pip install xmltodict).") from e
            try:
                return dict(xmltodict.parse(text))
            except Exception as ex:
                # fallback to sibling files if XML parse fails
                for ext in (".xml", ".XML"):
                    p_ext = path.with_suffix(ext)
                    if p_ext.exists():
                        return MPPP_Image._coerce_label_to_dict(p_ext, p_ext.read_text(encoding="utf-8", errors="ignore"))
                raise ex

        # Try PVL (PDS3)
        try:
            import pvl  # lazy import
        except ImportError as e:
            raise ImportError("pvl is required for PDS3 PVL label parsing (pip install pvl).") from e

        try:
            lbl = pvl.loads(text)
            return MPPP_Image._pvl_to_dict(lbl)
        except Exception as ex:
            # fallback to sibling labels
            for ext in (".lbl", ".LBL", ".lblx", ".LBLX", ".xml", ".XML"):
                p_ext = path.with_suffix(ext)
                if p_ext.exists():
                    return MPPP_Image._coerce_label_to_dict(
                        p_ext, p_ext.read_text(encoding="utf-8", errors="ignore")
                    )
            raise ex

    @staticmethod
    def _pvl_to_dict(obj: Any) -> Dict[str, Any]:
        """Recursively convert PVL structures to plain dict/list/scalars."""
        if isinstance(obj, dict):
            return {k: MPPP_Image._pvl_to_dict(v) for k, v in obj.items()}
        if hasattr(obj, "items"):
            return {k: MPPP_Image._pvl_to_dict(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [MPPP_Image._pvl_to_dict(v) for v in obj]
        return obj

    @staticmethod
    def _ensure_hw_or_hwc(arr: np.ndarray) -> np.ndarray:
        """Normalize to (H,W) or (H,W,C); transpose (C,H,W) → (H,W,C) when needed."""

        if arr.ndim == 2:
            return arr
        if arr.ndim == 3:
            if arr.shape[0] <= 8 and arr.shape[0] < arr.shape[1] and arr.shape[0] < arr.shape[2]:
                return np.transpose(arr, (1, 2, 0))
            return arr
        return np.squeeze(arr)

    # def _demosaic_image(arr: np.ndarray) -> np.ndarray:
    #     return colour_demosaicing.demosaicing_CFA_Bayer_Malvar2004( arr, 'RGGB' )
    
    # def _stack_image(arr: np.ndarray) -> np.ndarray:
    #     return  np.stack( [arr,arr,arr], axis=-1)

    @staticmethod
    def _get_label_scalar(label: Dict[str, Any], keys: list[str]) -> Optional[str]:
        """
        Fetch the first present key (case-insensitive).
        If the value is a list/tuple, return the first element.
        Always return as str for consistency.
        """
        def deep_get(dct: Dict[str, Any], k: str) -> Any:
            # case-insensitive single-level or nested via dot-notation
            parts = k.split(".")
            cur: Any = dct
            for part in parts:
                if not isinstance(cur, dict):
                    return None
                # exact first
                if part in cur:
                    cur = cur[part]
                    continue
                # case-insensitive match at this level
                target = part.lower()
                found = None
                for ck, cv in cur.items():
                    if isinstance(ck, str) and ck.lower() == target:
                        found = cv
                        break
                if found is None:
                    return None
                cur = found
            return cur

        for k in keys:
            v = deep_get(label, k)
            if v is not None:
                if isinstance(v, (list, tuple)) and len(v) > 0:
                    return str(v[0])
                return str(v)
        return None

def T_RM_from_cmod( GEOMETRIC_CAMERA_MODEL ):

    R_RM = R.from_quat( q_wxyz2xyzw(GEOMETRIC_CAMERA_MODEL['MODEL_TRANSFORM_QUATERNION'])).as_matrix()
    t_RM = GEOMETRIC_CAMERA_MODEL['MODEL_TRANSFORM_VECTOR'] 

    return R_RM,  t_RM 

def xyz_ned2enu( xyz ):
    return np.array( [ xyz[1], xyz[0], -xyz[2] ] )
        
def q_wxyz2xyzw( q_wxyz ):
     return np.array([ q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]] )

def find_ypr_from_R_ref( R_ref ):
    Q = R.from_matrix([[-1,0,0],[0,1,0],[0,0,-1]])
    ypr = ( Q.inv() * R.from_matrix(R_ref)).inv().as_euler("ZYX",degrees=1)
    return ypr

def find_R_ref_from_ypr( ypr ):
    Q = R.from_matrix([[-1,0,0],[0,1,0],[0,0,-1]])
    R_ref = (R.from_matrix( Q ) * R.from_euler('ZYX',ypr,degrees=1)).inv()
    return R_ref

def pad_image( im, pad = [0,0,0,0] ):

    dtype = im.dtype
    
    if len( im.shape ) == 3:
        im     = np.hstack( [ np.zeros( (im.shape[0],   pad[0] , 3), dtype=dtype), im, np.zeros( ( im.shape[0],  pad[1], 3), dtype=dtype), ] )
        im     = np.vstack( [ np.zeros( (pad[2]  , im.shape[1],  3), dtype=dtype), im, np.zeros( ( pad[3],  im.shape[1], 3), dtype=dtype), ] )
    else:
        im     = np.hstack( [ np.zeros( (im.shape[0],   pad[0]    ), dtype=dtype), im, np.zeros( ( im.shape[0],  pad[1]   ), dtype=dtype), ] )
        im     = np.vstack( [ np.zeros( (pad[2]  , im.shape[1],   ), dtype=dtype), im, np.zeros( ( pad[3],  im.shape[1]   ), dtype=dtype), ] )
    return im



def stack_image_with_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Stack an image and a binary mask into an array with an alpha channel.

    Args:
        image: HxW (grayscale) or HxWxC (color) numpy array.
               Must be uint8 or uint16 (bit depth taken from dtype).
        mask:  HxW boolean array (True/False).
               True => max intensity of dtype, False => 0.

    Returns:
        HxWx(C+1) array with alpha channel appended.
        - dtype matches input image dtype.
        - Alpha channel is scaled to full integer range of dtype.
    """
    if image.dtype not in (np.uint8, np.uint16):
        raise ValueError("Image must be uint8 or uint16.")

    if mask.shape != image.shape[:2]:
        raise ValueError("Mask must have same height/width as image.")

    # Maximum representable integer for dtype
    maxval = np.iinfo(image.dtype).max

    # Build alpha channel
    alpha = np.where(mask, maxval, 0).astype(image.dtype)

    # Expand grayscale image to HxWx1 if needed
    if image.ndim == 2:
        image = image[..., None]

    # Stack channels
    stacked = np.concatenate([image, alpha[..., None]], axis=2)
    return stacked

def undistort_isotropic(
    image: np.ndarray,
    K: np.ndarray,
    dist_coeffs: Sequence[float],
    f_strategy: str = "min",  # "min" (safer, less cropping) or "mean" (balanced FOV)
    center_projection: bool = True, 
    interpolation: int = cv2.INTER_LINEAR
    ) -> np.ndarray:

    """
    Undistort 'image' using OpenCV so that the rectified intrinsics have:
      - principal point at the image center, and
      - fx_new == fy_new (isotropic focal).

    Args:
        image: HxW or HxWxC (uint8/uint16/float) numpy array.
        K:     3x3 intrinsic matrix (fx, fy, cx, cy).
        dist_coeffs: Distortion coefficients as an iterable compatible with OpenCV
                     e.g., [k1, k2, p1, p2, k3] (or more if available).
        f_strategy:  "min" uses min(fx, fy) to avoid cropping; "mean" uses (fx+fy)/2.
        interpolation: OpenCV interpolation flag for remapping.

    Returns:
        undistorted image (same size as input).
    """
    if K.shape != (3, 3):
        raise ValueError("K must be a 3x3 intrinsic matrix.")

    h, w = np.shape(image)[:2]

    fx, fy = float(K[0, 0]), float(K[1, 1])
    if f_strategy == "mean":
        f_new = 0.5 * (fx + fy)
    else:
        f_new = min(fx, fy)


    # Center principal point
    if center_projection:
        cx_new, cy_new =[ w * 0.5,  h * 0.5]
    else: 
        cx_new, cy_new =[ float(K[0, 2]), float(K[1, 2])]

    K_new = np.array([[f_new, 0.0,   cx_new],
                      [0.0,   f_new, cy_new],
                      [0.0,   0.0,   1.0  ]], dtype=np.float64)
    
    # print( "old distortion coeficents", dist_coeffs )
    # print( "new distortion coeficents", 0 )

    # Identity rectification (no rotation), standard undistort-rectify maps
    R = np.eye(3, dtype=np.float64)
    map1, map2 = cv2.initUndistortRectifyMap(
        cameraMatrix=K.astype(np.float64),
        distCoeffs=np.asarray(dist_coeffs, dtype=np.float64),
        R=R,
        newCameraMatrix=K_new,
        size=(w, h),
        m1type=cv2.CV_32FC1
    )

    if np.issubdtype(image.dtype, np.floating): 
        image = image.astype(np.float32, copy=False)
        
    undistorted = cv2.remap(image, map1, map2, interpolation=interpolation, borderMode=cv2.BORDER_CONSTANT)

    return undistorted, K_new


# ---- Tiny FPN + Lite-ASPP ----
class TinyFPN(nn.Module):
    """Fuse last 3 stages (strides 8/16/32) to stride-4 feature."""
    def __init__(self, chs, out_ch=192):
        super().__init__()
        c2, c3, c4 = chs
        self.l2 = nn.Conv2d(c2, out_ch, 1, bias=False)
        self.l3 = nn.Conv2d(c3, out_ch, 1, bias=False)
        self.l4 = nn.Conv2d(c4, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.bn4 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.smooth = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
    def forward(self, c2, c3, c4):
        p4 = self.act(self.bn4(self.l4(c4)))
        p3 = self.act(self.bn3(self.l3(c3))) + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.act(self.bn2(self.l2(c2))) + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.smooth(p2)    # stride-8 → still stride-8 here (since c2 is stride-8)
        return p2

class LiteASPP(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.b1 = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.b2 = nn.Conv2d(in_ch, out_ch, 3, padding=2, dilation=2, bias=False)
        self.b3 = nn.Conv2d(in_ch, out_ch, 3, padding=4, dilation=4, bias=False)
        self.proj = nn.Conv2d(out_ch * 3, out_ch, 1, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.ReLU(inplace=True)
    def forward(self, x):
        y = torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)
        y = self.proj(y)
        return self.act(self.bn(y))


class ConvNeXtSeg(nn.Module):
    """
    ConvNeXt-Base backbone (features_only) + FPN + Lite-ASPP → 1ch logits (stride-1).
    """
    def __init__(self, pretrained=True, fpn_width=256):
        super().__init__()
        # Prefer IN22k weights; fallback to default if not available locally
        model_size = "tiny"
        names = [f"convnext_{model_size}.fb_in22k", f"convnext_{model_size}"]
        bb = None
        for n in names:
            try:
                bb = timm.create_model(n, features_only=True, pretrained=pretrained)
                break
            except Exception:
                pass
        if bb is None:
            raise RuntimeError("Could not create ConvNeXt. Install/upgrade timm or download weights.")
        self.bb = bb
        chs = self.bb.feature_info.channels()      # e.g., [128, 256, 512, 1024]
        self.pick = [-3, -2, -1]                  # use last 3 stages: 8/16/32
        self.fpn  = TinyFPN([chs[i] for i in self.pick], out_ch=fpn_width)
        self.aspp = LiteASPP(fpn_width, fpn_width)
        self.up2  = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)  # stride-8 → stride-4
        # head predicts at stride-4
        self.head = nn.Sequential(
            nn.Conv2d(fpn_width, fpn_width//2, 3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_width//2), nn.ReLU(inplace=True),
            nn.Conv2d(fpn_width//2, 1, 1)  # 1ch logits at stride-4
        )
        with torch.no_grad():
            if self.head[-1].bias is not None:
                self.head[-1].bias.fill_(math.log(0.55/0.45))

    def forward(self, x):
        feats = self.bb(x)
        c2, c3, c4 = [feats[i] for i in self.pick]  # stride-8/16/32
        p  = self.fpn(c2, c3, c4)                   # stride-8
        z  = self.aspp(p)                           # stride-8
        z  = self.up2(z)                            # stride-4
        logits_s4 = self.head(z)                    # Bx1xhxw (stride-4)
        logits = F.interpolate(logits_s4, size=x.shape[-2:], mode="bilinear", align_corners=False)  # full res
        return logits

def dilate_probability_map(prob_map: np.ndarray, threshold: float = 0.5,
                           kernel_size: int = 5, iterations: int = 1) -> np.ndarray:
    """
    Dilate a probability map before converting it into a binary mask.

    Args:
        prob_map (np.ndarray): 2D array of probabilities (float32, in [0,1]).
        thresh (float): Threshold for binary mask.
        kernel_size (int): Size of square structuring element for dilation.
        iterations (int): How many times to apply dilation.

    Returns:
        mask_dilated (np.ndarray): Dilated binary mask, dtype=uint8.
    """
    if prob_map.dtype != np.float32:
        prob_map = prob_map.astype(np.float32)

    # Threshold → binary mask (0/1)
    mask = (prob_map > threshold).astype(np.uint8)

    # Create structuring element (square kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    # Dilate
    mask_dilated = cv2.dilate(mask, kernel, iterations=iterations)

    return mask_dilated



@torch.no_grad()
def infer_segmentation(model, image, device=None, target_size=1648, threshold=0.5, fpn_width=256):
    """
    Runs ConvNeXt segmentation on one image. Returns (mask_bin, prob_map) in original HxW.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # -- load RGB image --
    try:
        img = image
    except:
        img = cv2.imread(image)
        if img is None:
            raise ValueError(f"Could not read {image}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    H0, W0 = img.shape[:2]

    # -- resize & pad to square (training size) --
    r = target_size / max(H0, W0)
    new_h, new_w = int(round(H0 * r)), int(round(W0 * r))
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_h, pad_w = target_size - new_h, target_size - new_w
    img_padded = np.pad(img_resized, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")

    # -- to torch fp32 and normalize (avoid numpy upcast to float64) --
    x = torch.from_numpy(img_padded).permute(2, 0, 1).unsqueeze(0).contiguous()  # 1x3xH xW, uint8
    x = x.to(device=device, dtype=torch.float32) / 255.0
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)
    mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=torch.float32).view(1,3,1,1)
    std  = torch.tensor(IMAGENET_STD,  device=device, dtype=torch.float32).view(1,3,1,1)
    x = (x - mean) / std

    # -- forward (keep fp32 to be safe on CPU too) --
    model.eval()
    logits = model(x)                       # 1x1xH xW (fp32)
    probs  = torch.sigmoid(logits)[0, 0]    # HxW (fp32)

    # -- unpad & resize back to original --
    probs = probs[:new_h, :new_w]
    probs = F.interpolate(probs[None, None], size=(H0, W0), mode="bilinear", align_corners=False)[0,0]
    prob_np = probs.cpu().numpy()

    mask_bin = (prob_np > threshold).astype(np.uint8)
    return mask_bin, prob_np


# load the model once
checkpoint_trained = "D:/code/MPPP/masks/m20/convnext_tiny_seg_best.pt"
model = ConvNeXtSeg( pretrained=False, fpn_width=256)
state = torch.load( checkpoint_trained, map_location="cpu")
model.load_state_dict(state["model"], strict=True)