import asyncio


from sfind.core.models import RetrieveRequest
from sfind.core.orchestrator import Orchestrator



async def main():
    file_path = "/Users/furqanshaikh/Documents/dev/sfind/images/tennis.jpg"


    # only bytes can be passed
    # xattr.setxattr(f=file_path,attr="users.test", value=b"some value")
    # try:
    #     print(xattr.getxattr(file_path, "users.test1"))
    # except OSError as e:
    #     print(f"Error getting extended attribute: {e}")
    # store_response = await VFSFileSystem().store_embedding(store_request=StoreRequest(
    #     file_path=file_path,model_id="1", embedding=b"embedding"
    # ))
    # print(store_response)
    # response = await VFSFileSystem().get_embedding(fetch_request=FetchRequest(file_path=file_path, model_id="1"))
    # print(response)
    def read_image_bytes(file_path: str):
        try:
            with open(file_path, "rb") as file:
                image_bytes = file.read()
            return image_bytes
        except FileNotFoundError:
            print(f"Error: The file '{file_path}' was not found.")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    # response = await CLIPModelFrontend().get_similarity_score(
    #     embed_text_request=EmbedTextRequest(text=["playing tennis"]),
    #     embed_image_request=EmbedImageRequest(name="cat", image_path=file_path)
    # )
    # response = await CLIPModelFrontend().embed_text(request=EmbedTextRequest(text=["photo of a cat"]))
    # response = await CLIPModelFrontend().get_similarity_score(embed_text_request=EmbedTextRequest(text=["photo of a cat"]),
    #                                                           embed_image_request=EmbedImageRequest(image_path=file_path))
    # print(response)

    response  = await Orchestrator()._retrieve(request=RetrieveRequest(
        file_types=[".jpg", ".jpeg"],
        path="/Users/furqanshaikh/Documents/dev/sfind/images",
        prompt="photo of a cat"
    ))
    print(response)
    # response = await VFSFileSystem().list_files("/Users/furqanshaikh/Documents/dev/sfind/images",
    #                                             file_types=[".jpg", ".jpeg"])
    # print(response)

if __name__ == "__main__":
    asyncio.run(main())
