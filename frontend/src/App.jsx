import { useContext, useState } from "react"
import CustomButton from "./components/buttons"
import "./App.css"
import ImageContainer from "./components/imagesContainer"
import HoodieEditor from "./components/hoodieEditor"
import ModelOptions from "./components/modelOptions"
import Configurator from "./components/configurator"
import { ActiveModelContext } from "./context/activeModelContext"
import ImageStyler from "./components/imageStyler"

function App() {
  const [images, setImages] = useState(null)
  const [loading, setLoading] = useState(false)
  const { activeModel } = useContext(ActiveModelContext)

  function get_images_handler(params) {
    setLoading(true)
    try {
      activeModel
        .func(params)
        .then((res) => {
          if (images && images.length > 0) setImages((prev) => [...res.img, ...prev])
          else setImages(res.img)
        })
        .catch((err) => console.log(err))
        .finally((_) => setLoading(false))
    } catch (err) {
      console.log(err.message)
      setLoading(false)
    }
  }

  return (
    <>
      <link rel='stylesheet' href='https://fonts.googleapis.com/icon?family=Material+Icons' />
      <div className='container'>
        <div style={{ display: "flex" }}>
          <ModelOptions />
          <ImageContainer images={images} loading={loading} />
          <Configurator getImagesHandler={get_images_handler} loading={loading} />
          <ImageStyler />
        </div>
        {/* 
        <CustomButton style={{ margin: "auto" }} onClick={get_images_handler} loading={loading}>
          Get Images
        </CustomButton> */}

        <HoodieEditor />
      </div>
    </>
  )
}

export default App
