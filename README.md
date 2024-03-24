Using Embedings (cool AI things) this tool scans cards and matches them to the cards in the database. This was developped for MTG where other card scanners exist, but assuming you have a GPU to enable the embedding calculation to go quickly, the process is much faster than other tools I've used, and seems to get much better results based on my testing. This method is also much easier to adapt to new cards, which is what made it easy to support lorcana, and easy to add new sets as soon as images of the cards are available.

Made possible thanks to [https://lorcast.com/](https://lorcast.com/). The APIs simple inteface and usability made developing this tool increadibly simple.

### Setup
1. Install python
    - https://www.python.org/downloads/
1. install podman and podman-compose
    - `pip3 install podman-compose`
2. Install Milvus
    - https://podman-desktop.io/
    - https://github.com/containers/podman/releases
    - `podman machine init`
    - `podman machine start`
    - `podman compose up -d`
4. install dependencies
    - `pip install requests pymilvus transformers opencv-python pillow pillow-avif-plugin`
5. install pytorch
    - `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
3. python dbcreate.py
    - this downloads the card images into onedrive, calculates the embedding, and adds that to the db
    - line 95 to 100 are commented out and can be used to only download a specific set, card name, or image type

### Run
1. open podman and start the mtg-card-scanner pod
2. run `python scan.py CSVFILENAME`
3. This will open a window showing your web cam.
4. If something is roughly card shaped it will zoom in on it, and run it through the DB
5. if there is a close enough match, the image will freeze with the card information overlayed on it
    - In this state you can either:
        - press escape to tell the camera to look again
        - press enter to add the standard card info to your csv along with price
        - press tab to add the foil card info to your csv along with price
        - press any arrow key to move to the next possible card
        - press an ascii character (letter, number, symbol) to refine the search to both cards with a name starting with that string, and collector set numbers (SET-123) starting with that string.
    - If keyboard input isn't be registered, click in the image window and try again
    - The multiple collector set numbers show the top (max 4) cards matching that embedding, most cards will be in that list but if they aren't use the text search or press escape and refocus the camera or move/turn the card
    - The zoomed image shows a zoomed and cropped lower left corner to more easilly see the set and number. This is not used in the embedding and is only shown for you.
6. while not showing card information to gracefully shutdown the program

