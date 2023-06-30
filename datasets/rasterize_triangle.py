import numpy as np
import os
import cv2
#import matplotlib.pyplot as plt
import math

def pointI2C(k_inv,cx,cy,point,depth): 
    x=depth*k_inv*(point[0]-cx)
    y=depth*k_inv*(point[1]-cy)
    xyz=np.array([x,y,depth])
    return xyz

def MINF3(x,y,z):
    return min(min(x,y),z)

def MAXF3(x,y,z):
    return max(max(x,y),z)

def  interpolation(pt,normal,normalPlane,depthmap,k_inv,cx,cy):
    h,w=depthmap.shape
    if pt[0]>0 and pt[0]<w and pt[1]>0 and pt[1]<h:
        m=np.inner(normalPlane,pointI2C(k_inv,cx,cy,pt,1))
        if m>0:
            depthmap[pt[1],pt[0]]=1/m
    return depthmap

def RasterizeTriangle(normal, normalPlane, point1, point2, point3, depthmap, k_inv, cx, cy):
    Y1 = 16*point1[1]
    Y2 = 16*point2[1]
    Y3 = 16*point3[1]
    X1 = 16*point1[0]
    X2 = 16*point2[0]
    X3 = 16*point3[0]
    #deltas
    DX12 = X1-X2
    DX23 = X2-X3
    DX31 = X3-X1
    DY12 = Y1-Y2
    DY23 = Y2-Y3
    DY31 = Y3-Y1
    #fixed-point deltas
    #???可能会有问题
    FDX12 = DX12 << 4
    FDX23 = DX23 << 4
    FDX31 = DX31 << 4
    FDY12 = DY12 << 4
    FDY23 = DY23 << 4
    FDY31 = DY31 << 4
    minx = int(MINF3(X1, X2, X3) + 0xF) >> 4
    maxx = int(MAXF3(X1, X2, X3) + 0xF) >> 4
    miny = int(MINF3(Y1, Y2, Y3) + 0xF) >> 4
    maxy = int(MAXF3(Y1, Y2, Y3) + 0xF) >> 4
    q=8
    minx &=~(q-1)
    miny &=~(q-1)
    # half-edge constants
    C1=DY12*X1-DX12*Y1
    C2=DY23*X2-DX23*Y2
    C3=DY31*X3-DX31*Y3

	#Correct for fill convention
    if DY12 < 0 or (DY12 == 0 and DX12 > 0):
        C1 +=1
    if DY23 < 0 or (DY23 == 0 and DX23 > 0):
        C2 +=1
    if DY31 < 0 or (DY31 == 0 and DX31 > 0):
        C3 +=1

    #Loop through blocks
    pixy = miny
    for y in range(miny,maxy,q):
        for x in range(minx,maxx,q):
            # Corners of block
            x0 = int(x) << 4
            x1 = int(x + q - 1) << 4
            y0 = int(y) << 4
            y1 = int(y + q - 1) << 4

			# Evaluate half-space functions
            a00 = C1 + DX12 * y0 - DY12 * x0 > 0
            a10 = C1 + DX12 * y0 - DY12 * x1 > 0
            a01 = C1 + DX12 * y1 - DY12 * x0 > 0
            a11 = C1 + DX12 * y1 - DY12 * x1 > 0
            a = (a00 << 0) | (a10 << 1) | (a01 << 2) | (a11 << 3)

            b00 = C2 + DX23 * y0 - DY23 * x0 > 0
            b10 = C2 + DX23 * y0 - DY23 * x1 > 0
            b01 = C2 + DX23 * y1 - DY23 * x0 > 0
            b11 = C2 + DX23 * y1 - DY23 * x1 > 0
            b = (b00 << 0) | (b10 << 1) | (b01 << 2) | (b11 << 3)

            c00 = C3 + DX31 * y0 - DY31 * x0 > 0
            c10 = C3 + DX31 * y0 - DY31 * x1 > 0
            c01 = C3 + DX31 * y1 - DY31 * x0 > 0
            c11 = C3 + DX31 * y1 - DY31 * x1 > 0
            c = (c00 << 0) | (c10 << 1) | (c01 << 2) | (c11 << 3)
            # Skip block when outside an edge
            if (a == 0x0 or b == 0x0 or c == 0x0):
                 continue

            nowpixy = pixy
            #Accept whole block when totally covered
            if (a == 0xF and b == 0xF and c == 0xF):
                for iy in range(q):
                    for ix in range(x,x+q):
                        #parser(ImageRef(ix,nowpixy))?
                        pt=np.array([ix,nowpixy])
                        depthmap=interpolation(pt,normal,normalPlane,depthmap,k_inv,cx,cy)

                    nowpixy +=1
            else: # Partially covered block
                CY1 = C1 + DX12 * y0 - DY12 * x0
                CY2 = C2 + DX23 * y0 - DY23 * x0
                CY3 = C3 + DX31 * y0 - DY31 * x0

                for iy in range(y, y + q):
                    CX1 = CY1
                    CX2 = CY2
                    CX3 = CY3

                    for ix in range(x, x + q):
                        if (CX1 > 0 and CX2 > 0 and CX3 > 0):
                            #parser(ImageRef(ix,nowpixy))
                            pt=np.array([ix,nowpixy])
                            depthmap=interpolation(pt,normal,normalPlane,depthmap,k_inv,cx,cy)
                            
                        CX1 -= FDY12
                        CX2 -= FDY23
                        CX3 -= FDY31

                    CY1 += FDX12
                    CY2 += FDX23
                    CY3 += FDX31

                    #++nowpixy;
                    nowpixy +=1             
        pixy += q	
    return depthmap