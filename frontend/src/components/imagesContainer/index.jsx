import { useContext } from "react"
import styles from "./imagesContainer.module.css"
import { SelectedImagesContext } from "../../context/selectedImagesContext"

function ImageContainer({ images, loading }) {
  const { setSelectedImages } = useContext(SelectedImagesContext)

  function onImageClickHandler(image) {
    setSelectedImages((prev) => [...prev, image])
  }

  if (!images || images.length == 0) {
    return <></>
  }
  return (
    <div style={{ margin: "auto", width: "fit-content", position: "relative" }}>
      {loading && (
        <div className={styles.loaderContainer}>
          <div className={styles.loader} />
        </div>
      )}
      <div className={styles.imageContainer}>
        {images.map((image, index) => (
          <img
            className={styles.image}
            key={index}
            onClick={() => onImageClickHandler(`data:image/png;base64,${image}`)}
            src={`data:image/png;base64,${image}`}
          />
        ))}
      </div>
    </div>
  )
}

export default ImageContainer
