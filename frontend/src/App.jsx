import { useContext, useState } from "react"
import CustomButton from "./components/buttons"
import "./App.css"
import ImageContainer from "./components/imagesContainer"
import HoodieEditor from "./components/hoodieEditor"
import ModelOptions from "./components/modelOptions"
import Configurator from "./components/configurator"
import { ActiveModelContext } from "./context/activeModelContext"

function App() {
  const [images, setImages] = useState(null)
  const [loading, setLoading] = useState(false)
  const { activeModel } = useContext(ActiveModelContext)

  function get_images_handler() {
    setLoading(true)
    activeModel
      .func()
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
        <div style={{ display: "flex" }}>
          <ModelOptions />
          <ImageContainer images={images} loading={loading} />
          <Configurator />
        </div>

        <CustomButton style={{ margin: "auto" }} onClick={get_images_handler} loading={loading}>
          Get Images
        </CustomButton>

        <HoodieEditor />
      </div>
    </>
  )
}

export default App
