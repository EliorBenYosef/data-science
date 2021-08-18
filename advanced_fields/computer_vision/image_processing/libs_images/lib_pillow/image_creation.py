# mode='L' forces the image to be parsed in the grayscale.


from PIL import Image


def show_white_image_with_vertical_lines(w, h):
    image_var = Image.new('L',(w,h),'white')
    pixel_matrix = image_var.load()

    for x in range(w):
        if x % 10 == 0:
            for y in range(h):
                pixel_matrix[x,y] = 0

    image_var.show()


def show_white_image_with_diagonal_line(n):
    image_var = Image.new('L',(n,n),'white')
    pixel_matrix = image_var.load()

    # less efficient
    # for x in range(n):
    #     for y in range(n):
    #         if x == y:
    #             pixel_matrix[x,y] = 0

    # more efficient
    for x in range(n):
        pixel_matrix[x,x] = 0

    image_var.show()


def show_white_image_with_circles(n):
    image_var = Image.new(mode='L', size=(n,n))
    pixel_matrix = image_var.load()

    for x in range(n):
        for y in range(n):
            pixel_matrix[x,y] = round((x-(n//2))**2+(y-(n//2))**2)% (n//2)

    image_var.show()


def show_psychedelic_image(n):
    image_var = Image.new('L',(n,n),'white')
    pixel_matrix = image_var.load()

    for x in range(n):
        for y in range(n):
            pixel_matrix[x,y] = ((x-(n//2))*(y-(n//2)))% (n//2)

    image_var.show()


def show_psychedelic_image_mine(n):
    image_var = Image.new('L',(n,n),'white')
    pixel_matrix = image_var.load()

    for x in range(n):
        for y in range(n):
            if y % 2 == 0:
                if x % 2 == 0:
                    pixel_matrix[x,y] = ((x+100)//2+(y-100)**2)% (n//2)
                else:
                    pixel_matrix[x,y] = ((x+100)**2+(y+100)//2)% (n//2)
            else:
                if x % 2 == 0:
                    pixel_matrix[x,y] = round((x-(n//2))**2+(y-(n//2))**2)% (n//2)
                else:
                    pixel_matrix[x,y] = round((x-(n//2))//2+(y-(n//2))//2)% (n//2)

    image_var.show()


def show_image_01(n):
    image_var = Image.new('L',(n,n),255)
    pixel_matrix = image_var.load()

    for i in range(n):
        for j in range(n):
            if (i+j)%2==0:
                pixel_matrix[i,j] = 0

    image_var.show()


def show_image_02(n):
    image_var = Image.new('L',(n,n),255)
    pixel_matrix = image_var.load()

    for i in range(n):
        for j in range(n):
            if i<=j:
                pixel_matrix[i,j] = 0

    image_var.show()


def show_image_03(n):
    image_var = Image.new('L',(n,n),255)
    pixel_matrix = image_var.load()

    for i in range(n):
        for j in range(n):
            pixel_matrix[i,j] = 20*i

    image_var.show()


show_white_image_with_vertical_lines(50,100)
show_white_image_with_diagonal_line(500)
show_white_image_with_circles(512)
show_psychedelic_image(512)
show_psychedelic_image_mine(512)
show_image_01(500)
show_image_02(500)
show_image_03(500)
