from flet import *

class CustomCheckBox(UserControl):
    def __init__(self, color, label='', selection_fill='#183588', size=25, stroke_width=2, animation=None, checked=False, font_size=17, pressed=None):
        super().__init__()
        self.selection_fill = selection_fill
        self.color = color
        self.label = label
        self.size = size
        self.stroke_width = stroke_width
        self.animation = animation
        self.checked = checked
        self.font_size = font_size
        self.pressed = pressed

    def _checked(self):
        self.check_box = Container(
            animate=self.animation,
            width=self.size, height=self.size,
            border_radius=(self.size / 2) + 5,
            bgcolor=self.CHECKED,
            content=Icon(icons.CHECK_ROUNDED, size=15),
        )
        return self.check_box

    def _unchecked(self):
        self.check_box = Container(
            animate=self.animation,
            width=self.size, height=self.size,
            border_radius=(self.size / 2) + 5,
            bgcolor=None,
            border=border.all(color=self.color, width=self.stroke_width),
            content=Container(),
        )
        return self.check_box

    def build(self):
        self.BG = '#041955'
        self.FG = '#3450a1'
        self.PINK = '#eb06ff'
        self.CHECKED = '#183588'

        if self.checked:
            return Column(controls=[
                Container(
                    on_click=lambda e: self.checked_check(e),
                    content=Row(
                        controls=[
                            self._checked(),
                            Text(self.label,
                                 font_family='poppins',
                                 size=self.font_size,
                                 weight=FontWeight.W_300),
                        ]
                    )
                )
            ])
        else:
            return Column(controls=[
                Container(on_click=lambda e: self.checked_check(e),
                          content=Row(
                              controls=[
                                  self._unchecked(),
                                  Text(self.label,
                                       font_family='poppins',
                                       size=self.font_size,
                                       weight=FontWeight.W_300),
                              ]
                          )
                          )
            ])

    def checked_check(self, e):
        print(self.checked)
        if not self.checked:
            self.checked = True
            self.check_box.border = None
            self.check_box.bgcolor = self.CHECKED
            self.check_box.content = Icon(icons.CHECK_ROUNDED, size=15)
            self.update()
        elif self.checked:
            self.checked = False
            self.check_box.bgcolor = None
            self.check_box.border = border.all(color=self.color, width=self.stroke_width)
            self.check_box.content.visible = False
            self.update()

        if self.pressed:
            self.run()

    def is_checked(self):
        return self.checked

    def run(self, *args):
        self.pressed(args)

def main(page: Page):
    print("Initializing page")  # Debug print

    BG = '#041955'
    FWG = '#97b4ff'
    FG = '#3450a1'
    PINK = '#eb06ff'

    circle = Stack(
        controls=[
            Container(
                width=100,
                height=100,
                border_radius=50,
                bgcolor='white12'
            ),
            Container(
                gradient=SweepGradient(
                    center=alignment.center,
                    start_angle=0.0,
                    end_angle=3,
                    stops=[0.5, 0.5],
                    colors=['#00000000', PINK],
                ),
                width=100,
                height=100,
                border_radius=50,
                content=Row(
                    alignment='center',
                    controls=[
                        Container(
                            padding=padding.all(5),
                            bgcolor=BG,
                            width=90,
                            height=90,
                            border_radius=50,
                            content=Container(
                                bgcolor=FG,
                                height=80,
                                width=80,
                                border_radius=40,
                                content=CircleAvatar(
                                    opacity=0.8,
                                    foreground_image_src=f"/images/ui.png"
                                ),
                            ),
                        ),
                    ],
                ),
            ),
        ],
    )

    def shrink(e):
        print("Shrink called")  # Debug print
        page_2.controls[0].width = 120
        page_2.controls[0].scale = transform.Scale(0.8, alignment=alignment.center_right)
        page_2.controls[0].border_radius = border_radius.only(
            top_left=35,
            top_right=0,
            bottom_left=35,
            bottom_right=0
        )
        page_2.update()

    def restore(e):
        print("Restore called")  # Debug print
        page_2.controls[0].width = 400
        page_2.controls[0].border_radius = 35
        page_2.controls[0].scale = transform.Scale(1, alignment=alignment.center_right)
        page_2.update()

    # List to store tasks
    tasks_list = []

    task_name_input = TextField(label='Task Name', width=300)

    def save_task():
        print("Task Saved!")  # Debug print
        new_task_name = task_name_input.value
        tasks_list.append(new_task_name)
        print(f"Tasks list: {tasks_list}")  # Debug print
        # Add the new task to the tasks column (if needed)
        tasks.controls.append(
            Container(
                height=70,
                width=400,
                bgcolor=BG,
                border_radius=25,
                padding=padding.only(left=20, top=25),
                content=CustomCheckBox(
                    color=PINK,
                    label=new_task_name
                )
            )
        )
        page.update()
        page.go('/')  # Navigate back to the main page

    create_task_view = Container(
        content=Column(
            controls=[
                Row(
                    alignment='end',
                    controls=[
                        Container(
                            on_click=lambda _: page.go('/'),
                            height=40,
                            width=40,
                            content=Text('x')
                        ),
                    ],
                ),
                Container(height=20),
                Text('Add a New Task', size=24, weight='bold'),
                task_name_input,
                ElevatedButton(
                    text='Save',
                    on_click=lambda _: save_task()
                )
            ]
        )
    )

    tasks = Column(
        height=400,
        scroll='auto',
    )
    for i in range(10):
        tasks.controls.append(
            Container(
                height=70,
                width=400,
                bgcolor=BG,
                border_radius=25,
                padding=padding.only(left=20, top=25),
                content=CustomCheckBox(
                    color=PINK,
                    label='Create interesting content!'
                )
            )
        )

    categories_card = Row(
        scroll='auto'
    )
    categories = ['Business', 'Family', 'Friends']
    for i, category in enumerate(categories):
        categories_card.controls.append(
            Container(
                border_radius=20,
                bgcolor=BG,
                width=170,
                height=110,
                padding=15,
                content=Column(
                    controls=[
                        Text('40 Tasks'),
                        Text(category),
                        Container(
                            width=160,
                            height=5,
                            bgcolor='white12',
                            border_radius=20,
                            padding=padding.only(right=i*30),
                            content=Container(
                                bgcolor=PINK,
                            ),
                        )
                    ]
                )
            )
        )

    first_page_contents = Container(
        content=Column(
            controls=[
                Row(
                    alignment='spaceBetween',
                    controls=[
                        Container(
                            on_click=lambda e: shrink(e),
                            content=Icon(icons.MENU)
                        ),
                        Row(
                            controls=[
                                Icon(icons.SEARCH),
                                Icon(icons.NOTIFICATIONS_OUTLINED)
                            ],
                        ),
                    ],
                ),
                Container(height=20),
                Text(value='What\'s up, Wahab!'),
                Text(value='CATEGORIES'),
                Container(
                    padding=padding.only(top=10, bottom=20),
                    content=categories_card
                ),
                Container(height=20),
                Text("TODAY'S TASKS"),
                Stack(
                    controls=[
                        tasks,
                        FloatingActionButton(
                            bottom=2,
                            right=20,
                            icon=icons.ADD,
                            on_click=lambda _: page.go('/create_task')
                        )
                    ]
                )
            ],
        ),
    )

    page_1 = Container(
        width=400,
        height=850,
        bgcolor=BG,
        border_radius=35,
        padding=padding.only(left=50, top=60, right=200),
        content=Column(
            controls=[
                Row(
                    alignment='end',
                    controls=[
                        Container(
                            border_radius=25,
                            padding=padding.only(top=13, left=13),
                            height=50,
                            width=50,
                            border=border.all(color='white', width=1),
                            on_click=lambda e: restore(e),
                            content=Text('<')
                        )
                    ]
                ),
                Container(height=20),
                circle,
                Text('Wahab\Naseer', size=32, weight='bold'),
                Container(height=25),
                Row(
                    controls=[
                        Icon(icons.FAVORITE_BORDER_SHARP, color='white60'),
                        Text('Templates', size=15, weight=FontWeight.W_300, color='white', font_family='poppins')
                    ]
                ),
                Container(height=5),
                Row(
                    controls=[
                        Icon(icons.CARD_TRAVEL, color='white60'),
                        Text('Templates', size=15, weight=FontWeight.W_300, color='white', font_family='poppins')
                    ]
                ),
                Container(height=5),
                Row(
                    controls=[
                        Icon(icons.CALCULATE_OUTLINED, color='white60'),
                        Text('Templates', size=15, weight=FontWeight.W_300, color='white', font_family='poppins')
                    ]
                ),
                Image(
                    src=f"/images/1.png",
                    width=300,
                    height=200,
                ),
                Text('Good', color=FG, font_family='poppins'),
                Text('Consistency', size=22)
            ]
        )
    )

    page_2 = Row(
        alignment='end',
        controls=[
            Container(
                width=400,
                height=850,
                bgcolor=FG,
                border_radius=35,
                animate=animation.Animation(600, AnimationCurve.DECELERATE),
                animate_scale=animation.Animation(400, curve='decelerate'),
                padding=padding.only(top=50, left=20, right=20, bottom=5),
                content=Column(
                    controls=[
                        first_page_contents
                    ]
                )
            )
        ]
    )

    container = Container(
        width=400,
        height=850,
        bgcolor=BG,
        border_radius=35,
        content=Stack(
            controls=[
                page_1,
                page_2,
            ]
        )
    )

    print("Setting up routes")  # Debug print

    pages = {
        '/': View(
            "/",
            [
                container
            ],
        ),
        '/create_task': View(
            "/create_task",
            [
                create_task_view
            ],
        )
    }

    def route_change(route):
        print(f"Route changed to {route}")  # Debug print
        page.views.clear()
        page.views.append(
            pages[page.route]
        )
        page.update()

    page.on_route_change = route_change

    print("Navigating to initial route")  # Debug print
    page.go(page.route)

app(target=main, assets_dir='assets')
