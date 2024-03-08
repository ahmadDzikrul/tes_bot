import discord
from discord.ext import commands
import requests
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix='$', intents=intents)

def pendeteksi(path_image):
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = load_model("keras_model.h5", compile=False)

    # Load the labels
    class_names = open("labels.txt", "r").readlines()

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open(path_image).convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", confidence_score)

    return class_name[2:]

@bot.event
async def on_ready():
    print(f'We have logged in as {bot.user}')

@bot.command()
async def hello(ctx):
    await ctx.send(f'Hi! I am a bot {bot.user}!')

@bot.command()
async def cara_mengurangi_sampah_dirumah(ctx):
    await ctx.send('''
- Memisahkan Sampah Sesuai Jenisnya
- Melakukan Zero Waste
- Membuat Pupuk dari Sampah Organik
- Membersihkan Tempat Sampah Setiap Hari
- Melakukan Daur Ulang Pada Sampah Anorganik
                   ''')
    
    

@bot.command()
async def heh(ctx, count_heh = 5):
    await ctx.send("he" * count_heh)

@bot.command()
async def coy(ctx, count_coy = 5):
    await ctx.send("coy" * count_coy)

def get_duck_image_url():    
    url = 'https://random-d.uk/api/random'
    res = requests.get(url)
    data = res.json()
    return data['url']


@bot.command('duck')
async def duck(ctx):
    '''Setelah kita memanggil perintah bebek (duck), program akan memanggil fungsi get_duck_image_url'''
    image_url = get_duck_image_url()
    await ctx.send(image_url)

@bot.command()
async def klasifikasi(ctx):
    data = ctx.message.attachments
    print(list(data)[-1])

    # kode untuk AI
    print('hasilnya adalah')
    response = requests.get(list(data)[-1])
    with open("image.jpg", "wb") as f:
        f.write(response.content)
    result = pendeteksi("image.jpg")

    await ctx.send(result)

bot.run("MTE1NzI2ODQ2ODgxOTgyNDc0MA.GSa0YR.xVsxWku_dUJUrabKkpwHoNcICzBcqrmGjJAfSk")