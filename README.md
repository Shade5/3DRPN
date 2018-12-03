# 3DRPN
Region proposal network that proposes 3D bounding boxes using two views of
the world[at 0 and 90 degrees]. 

This repo also contains to generate renders using ShapeNet data. The script creates the 3D world with random number of objects at different scales and locations and also with different lighting conditions. Images are rendered by moving the camera around the entire scene by fixed increments. At each angle of rotation of the camera, images are rendered at 3 different elevation angles. 


ToDo:

1] Code to generate depth maps 
2] Code to generate voxel representation
3] Add non-max suppression


