from manim import *

class encodeanim(Scene):
    def construct(self):
        # Create the text objects
        self.camera.background_color = "#ece6e2"
        text1 = Text("Lord", font_size=70)
        text2 = Text("Of", font_size=70)
        text3 = Text("The", font_size=70)
        text4 = Text("Rings", font_size=70)
        text_array = [text1, text2, text3, text4]
        number_aray = [431, 28, 9089, 7]
        
        # Create the VGroup of text objects
        text_objs = VGroup(*text_array)
        text_objs.arrange(RIGHT, buff=1.2)
        text_objs.set_color(BLACK)
        
        # Create a list of surrounding rectangles for each text object
        rects = [SurroundingRectangle(text, color=RED, buff=0.1) for text in text_array]
        
        numbers = [Text(str(i), color=BLUE, font_size=60) for i in number_aray]

        
        # Add the VGroup and surrounding rectangles to the scene
        group = VGroup(text_objs)
        self.play(Create(group), run_time=2)
        group2 = VGroup(*rects)
        self.play(Create(group2), run_time=2)
        
        arrows = [Arrow(UP, DOWN, color=RED) for _ in range(len(text_array))]
        

        for i, rect in enumerate(rects):
            arrow = arrows[i]
            number = numbers[i]
            self.play(
                arrow.animate.next_to(rect, DOWN, buff=0.1),
                run_time=0.8
            )
            self.play(
                number.animate.next_to(arrow, DOWN, buff=0.1),
                run_time=1
            )

        
        # # Animate the surrounding rectangles to slide over the corresponding text objects
        # for i, rect in enumerate(rects):
            
        #     # self.play(rect.animate.next_to(text_array[i], DOWN, buff=0.1), run_time=1)


