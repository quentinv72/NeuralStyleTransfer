# Neural Style Transfer

This is a web app that lets you generate images using the neural style transfer algorithm as exposed [here](https://arxiv.org/abs/1508.06576).

The user chooses an art image from our selection and uploads a picture of their choosing. Then the algorithm will generate a new image that applies the style of the art image to the content of the user's image.

For example using these two images

<img src="./images/dancing.jpg" width="256">
<img src='./images/krichner.jpg' width="256">

Will generate an image similar to this:

<img src='./images/kirchner_dancer.jpeg' width="256">

## Running on local machine

This web app has not yet been deployed. However, if you would like to test it on your local machine, then follow these instructions:

1. Clone the repository
2. Install requirements for Python

```
pip install -r requirements.txt
```

3. Install client packages using yarn

```
cd client && yarn install
```

4. Start the client server (make sure you are in the `/client` directory)

```
yarn start
```

5. Start API server. Go to another terminal window and run commands in `/api` directory

```
uvicorn main:app --reload
```

6. Finally go to http://localhost:3000 in your browser.
