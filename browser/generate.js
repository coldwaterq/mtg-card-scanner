import { pipeline, env, Tensor } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers';
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.webgpu.min.js';

// Since we will download the model from the Hugging Face Hub, we can skip the local model check
env.allowLocalModels = false;

// Reference the elements that we will need
const status = document.getElementById('status');
const setIdInp = document.getElementById('set-id');
const setSubmition = document.getElementById('set-submit');

function cosineSimilarity(query_embeds, database_embeds) {
    const EMBED_DIM = 512;

    const numDB = database_embeds.length / EMBED_DIM;
    const similarityScores = new Array(numDB);

    for (let i = 0; i < numDB; ++i) {
        const startOffset = i * EMBED_DIM;
        const dbVector = database_embeds.slice(startOffset, startOffset + EMBED_DIM);

        let dotProduct = 0;
        let normEmbeds = 0;
        let normDB = 0;

        for (let j = 0; j < EMBED_DIM; ++j) {
            const embedValue = query_embeds[j];
            const dbValue = dbVector[j];

            dotProduct += embedValue * dbValue;
            normEmbeds += embedValue * embedValue;
            normDB += dbValue * dbValue;
        }

        similarityScores[i] = dotProduct / (Math.sqrt(normEmbeds) * Math.sqrt(normDB));
    }

    return similarityScores;
}

// getCachedFile is not created, and uses nodejs stuff.
// get the mebeddings
// embeddings = getCachedFile(this.BASE_URL + 'audio-embeddings_52768-512_32bit.bin')
//                     .then((buffer) => {
//                         resolve(new Float32Array(buffer));
//                     })
//                     .catch(reject);

// Create a new object detection pipeline
status.textContent = 'Loading model...';
const features =  await pipeline('image-feature-extraction', 'Xenova/clip-vit-large-patch14', {
    device: 'webgpu'
});
status.textContent = 'Ready';

let pyodide = await loadPyodide({
    indexURL : "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/"
});
setSubmition.addEventListener("click", function(e) {
    var setId = setIdInp.value;
    fetch("https://api.scryfall.com/sets/"+setId)
        .then( response => response.json())
        .then( data => {
            getCards(setId, data["search_uri"]);
        })
});


const delay = ms => new Promise(res => setTimeout(res, ms));

function getCards(setId, url){
    fetch(url)
        .then( response => response.json())
        .then( async data => {
            let cards = {}
            var i = 0;
            for (let card of data["data"]){
                if (card.hasOwnProperty("card_faces")){
                    computeEmbed(
                        setId+"-"+card["collector_number"],
                        card["card_faces"][0]["image_uris"]["png"],
                        cards,
                        data["data"].length
                    )
                    computeEmbed(
                        setId+"-"+card["collector_number"],
                        card["card_faces"][1]["image_uris"]["png"],
                        cards,
                        data["data"].length
                    )
                } else {
                    computeEmbed(
                        setId+"-"+card["collector_number"],
                        card["image_uris"]["png"],
                        cards,
                        data["data"].length
                    )
                }
                if (i++ > 2){
                    break;
                }
            } 
            console.log(cards)
            var mime_type = "text/plain";

            
            while (Object.keys(cards).length < data["data"].length){
                await delay(5000);
                console.log(JSON.stringify(cards));
            }
            await delay(5000);
            var blob = new Blob([JSON.stringify(cards)], {type: mime_type});
            var url = window.URL.createObjectURL(blob);
            window.open(url, '_blank')
            
                
        })
    
    
}


// Detect objects in the image
async function computeEmbed(card,url,cards,slength) {
    const output = await features(url);
    console.log(output.tolist())
    cards[card]=output.tolist()[0]
    status.textContent = Object.keys(cards).length+"/"+slength
    alert(pyodide.runPython("print('Hello, world from the browser!')"));
    
    // scores = cosineSimilarity(output.data,embeddings)
}

// Render a bounding box and label on the image
function renderBox({ box, label }) {
    const { xmax, xmin, ymax, ymin } = box;

    // Generate a random color for the box
    const color = '#' + Math.floor(Math.random() * 0xFFFFFF).toString(16).padStart(6, 0);

    // Draw the box
    const boxElement = document.createElement('div');
    boxElement.className = 'bounding-box';
    Object.assign(boxElement.style, {
        borderColor: color,
        left: 100 * xmin + '%',
        top: 100 * ymin + '%',
        width: 100 * (xmax - xmin) + '%',
        height: 100 * (ymax - ymin) + '%',
    })

    // Draw label
    const labelElement = document.createElement('span');
    labelElement.textContent = label;
    labelElement.className = 'bounding-box-label';
    labelElement.style.backgroundColor = color;

    boxElement.appendChild(labelElement);
    imageContainer.appendChild(boxElement);
}