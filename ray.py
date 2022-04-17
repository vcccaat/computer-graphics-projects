import numpy as np
import math
from utils import *

"""
Core implementation of the ray tracer.  This module contains the classes (Sphere, Mesh, etc.)
that define the contents of scenes, as well as classes (Ray, Hit) and functions (shade) used in 
the rendering algorithm, and the main entry point `render_image`.

In the documentation of these classes, we indicate the expected types of arguments with a
colon, and use the convention that just writing a tuple means that the expected type is a
NumPy array of that shape.  Implementations can assume these types are preconditions that
are met, and if they fail for other type inputs it's an error of the caller.  (This might 
not be the best way to handle such validation in industrial-strength code but we are adopting
this rule to keep things simple and efficient.)
"""


class Ray:

    def __init__(self, origin, direction, start=0., end=np.inf):
        """Create a ray with the given origin and direction.

        Parameters:
          origin : (3,) -- the start point of the ray, a 3D point
          direction : (3,) -- the direction of the ray, a 3D vector (not necessarily normalized)
          start, end : float -- the minimum and maximum t values for intersections
        """
        # Convert these vectors to double to help ensure intersection
        # computations will be done in double precision
        self.origin = np.array(origin, np.float64)
        self.direction = np.array(direction, np.float64)
        self.start = start
        self.end = end


class Material:

    def __init__(self, k_d, k_s=0., p=20., k_m=0., k_a=None): 
        """Create a new material with the given parameters.

        Parameters:
          k_d : (3,) -- the diffuse coefficient
          k_s : (3,) or float -- the specular coefficient
          p : float -- the specular exponent
          k_m : (3,) or float -- the mirror reflection coefficient
          k_a : (3,) -- the ambient coefficient (defaults to match diffuse color)
        """
        self.k_d = k_d
        self.k_s = k_s
        self.p = p
        self.k_m = k_m
        self.k_a = k_a if k_a is not None else k_d


class Hit:

    def __init__(self, t, point=None, normal=None, material=None):
        """Create a Hit with the given data.

        Parameters:
          t : float -- the t value of the intersection along the ray
          point : (3,) -- the 3D point where the intersection happens
          normal : (3,) -- the 3D outward-facing unit normal to the surface at the hit point
          material : (Material) -- the material of the surface
        """
        self.t = t
        self.point = point
        self.normal = normal
        self.material = material

# Value to represent absence of an intersection
no_hit = Hit(np.inf)


class Sphere:

    def __init__(self, center, radius, material):
        """Create a sphere with the given center and radius.

        Parameters:
          center : (3,) -- a 3D point specifying the sphere's center
          radius : float -- a Python float specifying the sphere's radius
          material : Material -- the material of the surface
        """
        self.center = center
        self.radius = radius
        self.material = material

    def intersect(self, ray):
        """Computes the first (smallest t) intersection between a ray and this sphere.

        Parameters:
          ray : Ray -- the ray to intersect with the sphere
        Return:
          Hit -- the hit data
        """
        # TODO A4 implement this function
        diff = ray.origin - self.center
        d = ray.direction
        discriminant = d.dot(diff)** 2 - d.dot(d) * ((diff).dot(diff) - self.radius ** 2)
        t = np.inf
        if discriminant >= 0:
          t = (np.dot(-d,diff) - np.sqrt(discriminant))/ np.dot(d,d)
        if discriminant > 0 and t < ray.start: # two solutions
          t = (np.dot(-d,diff) + np.sqrt(discriminant))/ np.dot(d,d)

        if ray.start < t < ray.end:
          point = ray.origin + t * d
          normal = normalize(point - self.center)
          return Hit(t,point,normal, self.material)
       
        return no_hit





class Triangle:

    def __init__(self, vs, material):
        """Create a triangle from the given vertices.

        Parameters:
          vs (3,3) -- an arry of 3 3D points that are the vertices (CCW order)
          material : Material -- the material of the surface
        """
        self.vs = vs
        self.material = material

    def intersect(self, ray):
        """Computes the intersection between a ray and this triangle, if it exists.

        Parameters:
          ray : Ray -- the ray to intersect with the triangle
        Return:
          Hit -- the hit data
        """
        # TODO A4 implement this function
        vertice_a, vertice_b, vertice_c = self.vs[0], self.vs[1], self.vs[2]
        col_abc = vertice_a - vertice_b
        col_def = vertice_a - vertice_c
        col_ghi = ray.direction
        col_jkl = vertice_a - ray.origin
        a, b, c = col_abc[0], col_abc[1], col_abc[2]
        d, e, f = col_def[0], col_def[1], col_def[2]
        g, h, i = col_ghi[0], col_ghi[1], col_ghi[2]
        j, k, l = col_jkl[0], col_jkl[1], col_jkl[2]
        M = a*(e*i - h*f) + b*(g*f - d*i) + c*(d*h - e*g)
        if M == 0:
          return no_hit 
        t = -1 * (f*(a*k - j*b) + e*(j*c - a*l) + d*(b*l - k*c)) / M
        if t < ray.start or t > ray.end:
          return no_hit
        gamma = (i*(a*k - j*b) + h*(j*c - a*l) + g*(b*l - k*c)) / M
        if gamma < 0 or gamma > 1:
          return no_hit
        Beta = (j*(e*i - h*f) + k*(g*f - d*i) + l*(d*h - e*g)) / M
        if Beta < 0 or Beta > 1-gamma:
          return no_hit
        point = ray.origin + t * ray.direction
        # normal
        V = vertice_b - vertice_a
        W = vertice_c - vertice_a
        Nx = V[1]*W[2] - V[2]*W[1]
        Ny = V[2]*W[0] - V[0]*W[2]
        Nz = V[0]*W[1] - V[1]*W[0]
        normal = normalize(vec([Nx, Ny, Nz]))
        return Hit(t,point,normal, self.material)


class Camera:

    def __init__(self, eye=vec([0,0,0]), target=vec([0,0,-1]), up=vec([0,1,0]), 
                 vfov=90.0, aspect=1.0, ):
        """Create a camera with given viewing parameters.

        Parameters:
          eye : (3,) -- the camera's location, aka viewpoint (a 3D point)
          target : (3,) -- where the camera is looking: a 3D point that appears centered in the view
          up : (3,) -- the camera's orientation: a 3D vector that appears straight up in the view
          vfov : float -- the full vertical field of view in degrees
          aspect : float -- the aspect ratio of the camera's view (ratio of width to height)
        """
        self.eye = eye
        self.target = target
        self.up = up
        self.aspect = aspect
        self.vfov = vfov
        # self.d = 1 / np.tan(vfov/2)
        
        # TODO A4 implement this constructor to store whatever you need for ray generation

    def generate_ray(self, img_point):
        """Compute the ray corresponding to a point in the image.

        Parameters:
          img_point : (2,) -- a 2D point in [0,1] x [0,1], where (0,0) is the lower left 
                      corner of the image and (1,1) is the upper right
        Return:
          Ray -- The ray corresponding to that image location (not necessarily normalized)
        """
        # TODO A4 implement this function
        dVec = self.target - self.eye
        distance = np.sqrt(dVec[0]**2 + dVec[1]**2 + dVec[2]**2)
        W = normalize(-dVec)
        U = normalize(np.cross(self.up, W))
        V = normalize(np.cross(W, U))
        height = 2 * distance * np.tan(np.radians(self.vfov/2))
        width = self.aspect * height
   
        direction = - distance * W + (img_point[0] * width - width/2)  * U + (img_point[1] * height - height/2) * V 
        direction = direction
        return Ray(origin=self.eye, direction=direction)



class PointLight:

    def __init__(self, position, intensity):
        """Create a point light at given position and with given intensity

        Parameters:
          position : (3,) -- 3D point giving the light source location in scene
          intensity : (3,) or float -- RGB or scalar intensity of the source
        """
        self.position = position
        self.intensity = intensity

    def illuminate(self, ray, hit, scene):
        """Compute the shading at a surface point due to this light.

        Parameters:
          ray : Ray -- the ray that hit the surface
          hit : Hit -- the hit data
          scene : Scene -- the scene, for shadow rays
        Return:
          (3,) -- the light reflected from the surface
        """
        # TODO A4 implement this function
        v = normalize(-ray.direction)
        light = self.position - hit.point
        distance = light[0]**2 + light[1]**2 + light[2]**2
        l = normalize(light)
        half = normalize(v+l)
        k_d = hit.material.k_d
        k_s = hit.material.k_s
        I = self.intensity
        p = hit.material.p
        n = hit.normal
        L =  k_d * I * max(0,np.dot(n,l)) / distance + k_s * I * np.power(max(0,np.dot(n,half)),p)/ distance
     
        
     
        return L


class AmbientLight:

    def __init__(self, intensity):
        """Create an ambient light of given intensity

        Parameters:
          intensity (3,) or float: the intensity of the ambient light
        """
        self.intensity = intensity

    def illuminate(self, ray, hit, scene):
        """Compute the shading at a surface point due to this light.

        Parameters:
          ray : Ray -- the ray that hit the surface
          hit : Hit -- the hit data
          scene : Scene -- the scene, for shadow rays
        Return:
          (3,) -- the light reflected from the surface
        """
        # TODO A4 implement this function
        L = hit.material.k_a * self.intensity
        return L


class Scene:

    def __init__(self, surfs, bg_color=vec([0.2,0.3,0.5])):
        """Create a scene containing the given objects.

        Parameters:
          surfs : [Sphere, Triangle] -- list of the surfaces in the scene
          bg_color : (3,) -- RGB color that is seen where no objects appear
        """
        self.surfs = surfs
        self.bg_color = bg_color

    def intersect(self, ray):
        """Computes the first (smallest t) intersection between a ray and the scene.

        Parameters:
          ray : Ray -- the ray to intersect with the scene
        Return:
          Hit -- the hit data
        """
        # TODO A4 implement this function
        smallest_t = np.inf
        hit_result = no_hit
        for surf in self.surfs:
          h = surf.intersect(ray)
          if ray.start < h.t < ray.end and h.t < smallest_t:
            smallest_t = h.t
            hit_result = h
        # print(hit_result.t,'hit_result')
        return hit_result


MAX_DEPTH = 4

def shade(ray, hit, scene, lights, depth=0):
    """Compute shading for a ray-surface intersection.

    Parameters:
      ray : Ray -- the ray that hit the surface
      hit : Hit -- the hit data
      scene : Scene -- the scene
      lights : [PointLight or AmbientLight] -- the lights
      depth : int -- the recursion depth so far
    Return:
      (3,) -- the color seen along this ray
    When mirror reflection is being computed, recursion will only proceed to a depth
    of MAX_DEPTH, with zero contribution beyond that depth.
    """
    # TODO A4 implement this function
    
    
    color = vec([0,0,0])
    if depth >= MAX_DEPTH:
      return color

    for light in lights:
      # pointlight
      if type(light).__name__ == 'PointLight':
        l = normalize(light.position - hit.point)
        shadowRay = Ray(origin=hit.point,direction=l,start=0.0001)
        shadowHit = scene.intersect(shadowRay)
        if shadowHit != no_hit:  
          # only use ambient 
          continue
        else:
          # not block, normal shading
          color += light.illuminate(ray, hit, scene)
      else: #ambient
          color += light.illuminate(ray, hit, scene)  

    v = normalize(-ray.direction)
    reflectDirection = 2 * np.dot(hit.normal,v)*hit.normal - v
    reflectRay = Ray(origin=hit.point, direction=reflectDirection,start=0.0001)
    reflectHit = scene.intersect(reflectRay)  
    
    if reflectHit != no_hit: 
      color += hit.material.k_m * shade(reflectRay,reflectHit,scene,lights,depth+1)
    else:
      color += vec(scene.bg_color) * hit.material.k_m 


    return color




def render_image(camera, scene, lights, nx, ny):
    """Render a ray traced image.

    Parameters:
      camera : Camera -- the camera defining the view
      scene : Scene -- the scene to be rendered
      lights : Lights -- the lights illuminating the scene
      nx, ny : int -- the dimensions of the rendered image
    Returns:
      (ny, nx, 3) float32 -- the RGB image
    """
    # TODO A4 implement this function

    aspect = camera.aspect
    cam_img = np.zeros((ny,nx,3), np.float64)

    for x in range(nx):
        for y in range(ny):
            u = (x+0.5)/nx
            v = (y+0.5)/ny
            # print(x,y)
            ray = camera.generate_ray(np.array([u,v]))
            hit = scene.intersect(ray)
            
            if(hit.t < np.inf):
                cam_img[y,x] = shade(ray, hit, scene, lights)
                # cam_img[y,x] = hit.material.k_d
            
            else:
                cam_img[y,x] = scene.bg_color
           
    return cam_img[:,:,:]          
   
  
