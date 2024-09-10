<script lang="ts">
    import {Input} from "$lib/components/ui/input/index.ts";
    import {Label} from "$lib/components/ui/label/index.ts";
    import {
        env,
        AutoModel,
        AutoProcessor,
        RawImage,
    } from "@huggingface/transformers";
    import JSZip from "jszip";
    import fileSaver from "file-saver";
    import {Button} from "$lib/components/ui/button";
    const {saveAs} = fileSaver;

    let dragging: boolean = false;
    $: dragging_style = dragging ? "border-primary" : " border-dashed";

    let modelRef: AutoModel;
    let processorRef: AutoProcessor;
    let images: string[] = [];
    let isProcessing: boolean = false;

    let processedImages: string[] = [];
    let isDownloadReady: boolean = false;
    let prevImages: string[] = [];
    let prevProcessed: string[] = [];

    $: {
        const loadModel = async () => {
            try {
                if (!navigator.gpu) {
                    throw new Error("WebGPU is not supported in this browser.");
                }
                const model_id = "Xenova/modnet";
                env.backends.onnx.wasm.proxy = false;
                modelRef ??= await AutoModel.from_pretrained(model_id, {
                    device: "webgpu",
                });
                processorRef ??= await AutoProcessor.from_pretrained(model_id);
                console.log("Model loaded successfully.");
            } catch (err) {
                console.error(err);
            }
        };
        loadModel();
    }

    function handleFiles() {
        const fileList = this.files; /* now you can work with the file list */
        console.log(fileList);
        images = [
            ...prevImages,
            ...Array.from(fileList).map((file: File) =>
                URL.createObjectURL(file),
            ),
        ];
    }

    const removeImage = (index: number) => {
        images = prevImages.filter((_, i) => i !== index);
        processedImages = prevProcessed.filter((_, i) => i !== index);
    };

    const processImages = async () => {
        isProcessing = true;
        processedImages = [];
        const model = modelRef;
        const processor = processorRef;

        for (let i = 0; i < images.length; ++i) {
            // Load image
            const img = await RawImage.fromURL(images[i]);

            // Pre-process image
            const {pixel_values} = await processor(img);

            // Predict alpha matte
            const {output} = await model({input: pixel_values});

            const maskData = (
                await RawImage.fromTensor(
                    output[0].mul(255).to("uint8"),
                ).resize(img.width, img.height)
            ).data;

            // Create new canvas
            const canvas = document.createElement("canvas");
            canvas.width = img.width;
            canvas.height = img.height;
            const ctx = canvas.getContext("2d");

            // Draw original image output to canvas
            ctx?.drawImage(img.toCanvas(), 0, 0);

            // Update alpha channel
            const pixelData = ctx?.getImageData(0, 0, img.width, img.height);
            for (let i = 0; i < maskData.length; ++i) {
                pixelData!.data[4 * i + 3] = maskData[i];
            }
            ctx?.putImageData(pixelData!, 0, 0);
            processedImages = [...prevProcessed, canvas.toDataURL("image/png")];
            console.log(processedImages);
        }

        isProcessing = false;
        isDownloadReady = true;
    };

    const downloadAsZip = async () => {
        const zip = new JSZip();
        const promises = images.map(
            (image, i) =>
                new Promise((resolve) => {
                    const canvas = document.createElement("canvas");
                    const ctx = canvas.getContext("2d");

                    const img = new Image();
                    img.src = processedImages[i] || image;

                    img.onload = () => {
                        canvas.width = img.width;
                        canvas.height = img.height;
                        ctx?.drawImage(img, 0, 0);
                        canvas.toBlob((blob) => {
                            if (blob) {
                                zip.file(`image-${i + 1}.png`, blob);
                            }
                            resolve(null);
                        }, "image/png");
                    };
                }),
        );

        await Promise.all(promises);

        const content = await zip.generateAsync({type: "blob"});
        saveAs(content, "images.zip");
    };

    const clearAll = () => {
        images = [];
        processedImages = [];
        isDownloadReady = false;
    };

    const copyToClipboard = async (url) => {
        try {
            // Fetch the image from the URL and convert it to a Blob
            const response = await fetch(url);
            const blob = await response.blob();

            // Create a clipboard item with the image blob
            const clipboardItem = new ClipboardItem({[blob.type]: blob});

            // Write the clipboard item to the clipboard
            await navigator.clipboard.write([clipboardItem]);

            console.log("Image copied to clipboard");
        } catch (err) {
            console.error("Failed to copy image: ", err);
        }
    };

    const downloadImage = (url: string) => {
        const link = document.createElement("a");
        link.href = url;
        link.download = "image.png";
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };
</script>

<main class="flex flex-col items-center justify-center gap-4 px-6 py-12 sm:px-8 mx-auto">
    <h1 class="text-4xl font-bold text-center mb-2">Welcome to Local RmBG</h1>
<h2 class="text-lg font-semibold mb-2 text-center">
          In-browser background removal, powered by{" "}
          <a
            class="underline"
            target="_blank"
            href="https://github.com/xenova/transformers.js"
          >
            🤗 Transformers.js
          </a>
        </h2>
    <div class="grid w-full max-w-sm items-center gap-1.5">
        <Label for="picture">Upload an image</Label>
        <Input on:change={handleFiles} on:drop={handleFiles} on:dragover={dragging = true} on:dragleave={dragging = false} id="picture" type="file" accept="image/png, image/jpeg, image/jpg" multiple
            class="hover:border-primary {dragging_style}" />
        <p>{dragging_style}</p>
    </div>
    <Button on:click={processImages}>Process</Button>
    {#if isDownloadReady}
    <Button on:click={downloadAsZip} variant="outline">Download as zip</Button>
    {/if}
    <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
        {#each images as src, index (index)}
        <div class="relative group">
            <img src={processedImages[index] || src} alt={`Image ${index + 1}`}
                class="rounded-lg object-cover w-full h-48" />
            {#if processedImages[index]}
            <div
                class="absolute inset-0 bg-black bg-opacity-70 opacity-0 group-hover:opacity-100 transition-opacity duration-300 rounded-lg flex items-center justify-center">
                <button on:click={()=>
                    copyToClipboard(processedImages[index] || src)}
                    class="mx-2 px-3 py-1 bg-white text-gray-900 rounded-md hover:bg-gray-200 transition-colors
                    duration-200
                    text-sm"
                    aria-label={`Copy image ${index + 1} to clipboard`}
                    >
                    Copy 
                </button>
                <button on:click={()=>
                    downloadImage(processedImages[index] || src)}
                    class="mx-2 px-3 py-1 bg-white text-gray-900 rounded-md hover:bg-gray-200 transition-colors
                    duration-200
                    text-sm"
                    aria-label={`Download image ${index + 1}`}
                    >
                    Download
                </button>
            </div>
            {/if}
            <button on:click={()=> removeImage(index)}
                class="absolute top-2 right-2 bg-black bg-opacity-50 text-white w-6 h-6 rounded-full flex items-center
                justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-300 hover:bg-opacity-70"
                aria-label={`Remove image ${index + 1}`}
                >
                &#x2715;
            </button>
        </div>
        {/each}
    </div>
</main>
