import { useContext, useState } from "react"
import styles from "./styleTransferPane.module.css"
import Box from "@mui/material/Box"
import Modal from "@mui/material/Modal"
import CustomButton from "../buttons"
import { styleImages, trialImg } from "../../../utils/style_images"
import { use_nst } from "../../../utils/generate_images"
import { nst_res } from "../../../utils/testImage"
import { SelectedImagesContext } from "../../context/selectedImagesContext"

const style = {
  position: "absolute",
  top: "50%",
  left: "50%",
  transform: "translate(-50%, -50%)",
  width: "fit-content",
  bgcolor: "#252525",
  border: "none",
  outline: "none",
  boxShadow: 24,
  p: 4,
}

function StyleTransferPane() {
  const [styleImage, setStyleImage] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [open, setOpen] = useState(false)
  const [nstResult, setNstResult] = useState(null)
  const handleOpen = () => setOpen(true)
  const handleClose = () => setOpen(false)

  const { setActiveImage, activeImage } = useContext(SelectedImagesContext)

  function styleImageSelected(img) {
    setStyleImage(img)
    // if (image != null && image != undefined) handleOpen()
    handleOpen()
  }
  function nstActivate() {
    if (activeImage) {
      setIsLoading(true)
      use_nst({ style: styleImage, content: activeImage.replace("data:image/png;base64,", "") })
        .then((res) => {
          console.log(res)
          setNstResult(res.image)
        })
        .catch((err) => console.error(err.message))
        .finally((_) => setIsLoading(false))
    }
  }
  function selectNstResult() {
    setActiveImage(`data:image/png;base64,${nstResult}`)
    handleClose()
  }

  return (
    <>
      <div>
        <p style={{ letterSpacing: 1.5 }}>Choose any of the image styles listed below, to convert your image to that style!</p>

        <div style={{ display: "flex", flexWrap: "wrap", justifyContent: "center", gap: 10 }}>
          {styleImages.map((img) => (
            <img
              onClick={() => styleImageSelected(img)}
              key={img.id}
              className={styles.styleImage}
              src={img.src}
              width={100}
              height={100}
            />
          ))}
        </div>
      </div>
      <Modal open={open} onClose={handleClose}>
        <Box sx={style}>
          <h2 style={{ marginTop: 0 }}>Style Transfer</h2>
          <div style={{ display: "flex", justifyContent: "center", alignItems: "center", gap: 20 }}>
            {styleImage && <img src={styleImage.src} width={200} height={200} />}
            <CustomButton disabled={!activeImage && !styleImage} loading={isLoading} onClick={nstActivate}>
              Mix
            </CustomButton>
            {activeImage ? <img src={activeImage} width={200} height={200} /> : "Please select an Image first!"}
            {/* <img src={trialImg} width={200} height={200} /> */}
          </div>
          <div style={{ margin: "10px", height: 200, display: "flex", justifyContent: "center" }}>
            {nstResult && (
              <img
                className={styles.nstResult}
                width={200}
                src={`data:image/png;base64,${nstResult}`}
                onClick={selectNstResult}
              />
            )}
          </div>
        </Box>
      </Modal>
    </>
  )
}

export default StyleTransferPane
