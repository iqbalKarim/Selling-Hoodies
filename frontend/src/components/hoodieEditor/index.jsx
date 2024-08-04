import { forwardRef, useRef, useState, useContext, useEffect } from "react"
import styles from "./hoodieEditor.module.css"
import ReactToPrint from "react-to-print"
import { SelectedImagesContext } from "../../context/selectedImagesContext"

import { makeMoveable, Rotatable, Draggable, Scalable } from "react-moveable"
import MoveableHelper from "moveable-helper"
import _uniqueId from "lodash/uniqueId"
import uniqueId from "lodash/uniqueId"
import { cloneDeep } from "lodash"

const Moveable = makeMoveable([Draggable, Scalable, Rotatable])

function HoodieEditor() {
  let componentRef = useRef()

  return (
    <div>
      <ReactToPrint trigger={() => <button>Submit design!</button>} content={() => componentRef} />
      <ComponentToPrint ref={(el) => (componentRef = el)} />
    </div>
  )
}

const ComponentToPrint = forwardRef((props, ref) => {
  const [updateId, setUpdateId] = useState(uniqueId())
  const [styleObj, setStyleObj] = useState({ borderRadius: "50px", height: "150px", width: "100px" })
  const [activeImage, setActiveImage] = useState(null)
  const [backgroundColor, setBackgroundColor] = useState("black")

  const { selectedImages, setSelectedImages } = useContext(SelectedImagesContext)

  function onChangeAttributeHandler(attr, value) {
    switch (attr) {
      case "borderRadius":
        // if (value) {
        //   let index = activeImage?.imageIndex
        //   const temp = cloneDeep(styleObjs)
        //   temp[index].styleObj = { ...temp[index].styleObj, borderRadius: `${value}` }
        //   setStyleObjs(temp)
        // } else {
        //   let index = activeImage?.imageIndex
        //   const temp = cloneDeep(styleObjs)
        //   temp[index].styleObj = { ...temp[index].styleObj, borderRadius: "" }
        //   setStyleObjs(temp)
        // }
        // break
        if (value) setStyleObj((prev) => ({ ...prev, borderRadius: `${value}` }))
        else setStyleObj((prev) => ({ ...prev, borderRadius: "" }))
        break
      case "width":
        if (value) setStyleObj((prev) => ({ ...prev, width: `${value}` }))
        else setStyleObj((prev) => ({ ...prev, width: "" }))
        break
      case "height":
        if (value) setStyleObj((prev) => ({ ...prev, height: `${value}` }))
        else setStyleObj((prev) => ({ ...prev, height: "" }))
        break
      default:
        break
    }
  }

  function selectImage(imageIndex, image) {
    //   setActiveImage({ imageIndex, ...image })
    //   setStyleObj({ ...image?.styleObj })
  }

  // useEffect(() => {
  //   console.log(activeImage)
  //   if (activeImage?.imageIndex) {
  //     let index = activeImage.imageIndex
  //     let newSelectedImages = cloneDeep(selectedImages)
  //     newSelectedImages[index].styleObj = { ...activeImage.styleObj }
  //     setSelectedImages(newSelectedImages)
  //   }
  // }, [activeImage])
  // useEffect(() => {
  //   setStyleObjs((prev) => [...prev, { borderRadius: "50px", height: "100px", width: "100px" }])
  // }, [selectedImages])

  return (
    <div style={{ display: "flex" }}>
      <div style={{ background: backgroundColor }} className={`editContainer ${styles.editorContainer}`} ref={ref}>
        <div key={updateId}>
          {selectedImages?.map((img, index) => {
            // setStyleObjs((prev) => [...prev, { borderRadius: "50px", height: "150px", width: "100px" }])
            return (
              <MovableImage
                activeImage={activeImage}
                key={index}
                imageIndex={index}
                image={img}
                styleObj={styleObj}
                // styleObjs={styleObjs}
                onClickHandler={selectImage}
              />
            )
          })}
        </div>
        <img src='./cutout3.png' className={styles.hoodiePattern} />
      </div>

      <div className={styles.imageStyleOptions}>
        <p className={styles.sectionHeading}>Image Styles</p>

        <p className={styles.inputContainer}>
          <label>Border Radius</label>
          <input value={styleObj.borderRadius} onChange={(e) => onChangeAttributeHandler("borderRadius", e.target.value)} />
          {/* <input
            value={activeImage?.styleObj?.borderRadius || 0}
            onChange={(e) => {
              if (activeImage && activeImage.styleObj) onChangeAttributeHandler("borderRadius", e.target.value)
            }}
          /> */}
        </p>
        <p className={styles.inputContainer}>
          <label>Height</label>
          <input value={styleObj.height} onChange={(e) => onChangeAttributeHandler("height", e.target.value)} />
        </p>
        <p className={styles.inputContainer}>
          <label>Width</label>
          <input value={styleObj.width} onChange={(e) => onChangeAttributeHandler("width", e.target.value)} />
        </p>
        <p className={styles.inputContainer}>
          <label>Colors</label>
          <span style={{ background: "black" }} className={styles.colorSelector} onClick={() => setBackgroundColor("black")} />
          <span style={{ background: "teal" }} className={styles.colorSelector} onClick={() => setBackgroundColor("teal")} />
          <span
            style={{ background: "#0a1628" }}
            className={styles.colorSelector}
            onClick={() => setBackgroundColor("#0a1628")}
          />
          <span
            style={{ background: "#a38063" }}
            className={styles.colorSelector}
            onClick={() => setBackgroundColor("#a38063")}
          />
          <span
            style={{ background: "#4c4c4c" }}
            className={styles.colorSelector}
            onClick={() => setBackgroundColor("#4c4c4c")}
          />
          <span
            style={{ background: "#1f2f20" }}
            className={styles.colorSelector}
            onClick={() => setBackgroundColor("#1f2f20")}
          />
        </p>
      </div>
    </div>
  )
})

function MovableImage({ image, imageIndex, styleObj, styleObjs, onClickHandler, activeImage }) {
  const { setSelectedImages } = useContext(SelectedImagesContext)

  const [helper] = useState(() => {
    return new MoveableHelper()
  })

  function doubleClickHandler(img, index) {
    setSelectedImages((prev) => {
      prev.splice(index, 1)
      return [...prev]
    })
  }
  const uniq = _uniqueId("image-")
  return (
    <div className={styles.movableImageContainer}>
      <div id={uniq} className={styles.movableImage}>
        <img
          key={styleObj}
          onClick={() => onClickHandler(imageIndex, image)}
          style={styleObj}
          // style={image.styleObj}
          width={100}
          // className={activeImage?.imageIndex == imageIndex ? styles.activeImage : ""}
          src={image.image}
          onDoubleClick={() => doubleClickHandler(image, imageIndex)}
        />
      </div>

      <Moveable
        className={styles.movable_custom}
        target={`#${uniq}`}
        draggable={true}
        scalable={true}
        // keepRatio={true}
        rotatable={true}
        onDragStart={helper.onDragStart}
        onDrag={helper.onDrag}
        onScaleStart={helper.onScaleStart}
        onScale={helper.onScale}
        onRotateStart={helper.onRotateStart}
        onRotate={helper.onRotate}
      />
    </div>
  )
}

export default HoodieEditor
