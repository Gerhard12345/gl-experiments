# import os

# os.environ["KIVY_GL_BACKEND"] = "angle_sdl2"

__version__ = "2.0"

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics import Callback
from kivy.graphics import Rectangle
from kivy.graphics.texture import Texture
from kivy.graphics import Color
from kivy.graphics.cgl import cgl_get_backend_name

from kivy.uix.boxlayout import BoxLayout

from kivy.uix.widget import Widget


from .drawing.openglrenderer import OpenGLRenderer
from .scenes.scene import Scene1, Scene2, Scene3
from .objects.camera import Camera


class MyWindow(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scene = Scene3()
        self.scene.setCamera(Camera(eye=[-10, 13, 20], at=[0, 6, 12], up=[0, 1, 0]))
        self.renderer = OpenGLRenderer(4)
        with self.canvas:
            self.rectangle = Rectangle()
            self.rectangles = [Rectangle() for i in range(self.scene.n_lights)]
            self.cb = Callback()
        self.i = 0
        self.game_has_started = False
        Clock.schedule_interval(self.update_glsl, 1.0 / 60.0)

    def change_light_component(self, component, change_val, light_index):
        light = self.scene.get_unidirectional_light_position(int(light_index))
        light[component] += change_val
        self.scene.set_unidirectional_light_position(light, int(light_index))

    def on_touch_move(self, touch):
        super().on_touch_move(touch)
        if self.collide_point(*touch.pos):
            self.scene.camera.rotate_phi(touch.dx)
            self.scene.camera.rotate_theta(touch.dy)

    def on_touch_down(self, touch):
        super().on_touch_down(touch)
        if touch.is_double_tap and self.collide_point(*touch.pos):
            self.scene.camera.set_lookat_position(touch.dx)
            self.scene.camera.set_lookat_position(touch.dy)

    def start(self):
        self.game_has_started = True

    def update_glsl(self, _instruction):
        self.scene.update()
        self.renderer.update(self.scene)
        pixels = self.renderer.scene_renderer.colorToBuffer()
        self.rectangle.texture.blit_buffer(pixels, colorfmt="rgba", bufferfmt="ubyte")
        pixels = self.renderer.shadow_renderer.depthToBuffer()
        for i in range(self.scene.n_lights):
            self.rectangles[i].texture.blit_buffer((pixels[i]).flatten().tobytes(), colorfmt="red", bufferfmt="float")
        self.cb.ask_update()

    def on_size(self, sender, value):
        pos_x = 40 + int(0.15 * value[1])
        value2 = (int(value[0] - 20 - pos_x), int(value[1] - 40))
        self.rectangle.size = value2
        self.rectangle.pos = (self.pos[0] + pos_x, self.pos[1] + 20)
        self.rectangle.texture = Texture.create(size=value2, colorfmt="rgba")
        relative_size_of_shadow = 0.15
        spacing_between_shadow_images = 0.1
        total_width_boxes = relative_size_of_shadow * value[1] * self.scene.n_lights + relative_size_of_shadow * value[
            1
        ] * (self.scene.n_lights - 1) * (spacing_between_shadow_images)
        for j in range(self.scene.n_lights):
            base_size = int(value[1] * relative_size_of_shadow)
            rect_sizes = (base_size, base_size)
            buff_sizes = (6 * base_size, 6 * base_size)
            self.rectangles[j].size = rect_sizes
            self.rectangles[j].pos = (
                20,
                (self.pos[1] + value[1] - 20 - total_width_boxes)
                + (j) * (1 + spacing_between_shadow_images) * rect_sizes[1],
            )
            self.rectangles[j].texture = Texture.create(size=buff_sizes, colorfmt="rgba")

        if self.renderer is not None:
            self.renderer.shadow_renderer.resize(buff_sizes)
            self.renderer.scene_renderer.resize(value2)
            self.renderer.scene_renderer.set_shadow_size(buff_sizes)


class MyLargeWindow(BoxLayout):
    pass


class MainApp(App):
    def build(self):
        self.widget = MyLargeWindow()
        return self.widget


if __name__ == "__main__":
    gui = MainApp()
    gui.run()
