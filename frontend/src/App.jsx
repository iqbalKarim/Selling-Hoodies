import { useState } from "react"
import { get_image } from "../utils/generate_images"
import CustomButton from "./components/buttons"
import "./App.css"
import ImageContainer from "./components/imagesContainer"
import HoodieEditor from "./components/hoodieEditor"

function App() {
  const [images, setImages] = useState(null)
  const [loading, setLoading] = useState(false)

  function get_images_handler() {
    setLoading(true)
    get_image()
      .then((res) => {
        if (images && images.length > 0) setImages((prev) => [...res.img, ...prev])
        else setImages(res.img)
      })
      .catch((err) => console.log(err))
      .finally((_) => setLoading(false))
  }

  return (
    <>
      <div className='container'>
        <ImageContainer images={images} loading={loading} />

        <CustomButton style={{ margin: "auto" }} onClick={get_images_handler} loading={loading}>
          Get Images
        </CustomButton>

        <HoodieEditor />
      </div>
    </>
  )
}

export default App
