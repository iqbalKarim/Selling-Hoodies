import { forwardRef, useRef, useState, useContext, useEffect } from "react"
import styles from "./hoodieEditor.module.css"
import ReactToPrint from "react-to-print"
import { SelectedImagesContext } from "../../context/selectedImagesContext"

import { makeMoveable, Rotatable, Draggable, Scalable } from "react-moveable"
import MoveableHelper from "moveable-helper"
import _uniqueId from "lodash/uniqueId"

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
  const { selectedImages } = useContext(SelectedImagesContext)

  return (
    <div className={`editContainer ${styles.editorContainer}`} ref={ref}>
      {selectedImages?.map((img, index) => (
        <MovableImage key={index} imageIndex={index} image={img} />
      ))}
      <img src='./cutout3.png' className={styles.hoodiePattern} />

      {/* <svg viewBox='0 0 500 500'>{HoodieSVG}</svg> */}
    </div>
  )
})

function MovableImage({ image, imageIndex }) {
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
        <img width={100} src={image} onDoubleClick={() => doubleClickHandler(image, imageIndex)} />
      </div>

      <Moveable
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
