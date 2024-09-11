import { forwardRef, useRef, useState, useContext, useEffect } from "react"
import styles from "./hoodieEditor.module.css"
import ReactToPrint from "react-to-print"
import { SelectedImagesContext } from "../../context/selectedImagesContext"

import { makeMoveable, Rotatable, Draggable, Scalable } from "react-moveable"
import MoveableHelper from "moveable-helper"
import _uniqueId from "lodash/uniqueId"
import CustomSlider from "../customSlider"

const Moveable = makeMoveable([Draggable, Scalable, Rotatable])

function HoodieEditor() {
  const { setHoodieBackground } = useContext(SelectedImagesContext)
  const [text, setText] = useState("Slipknot")
  const [textStyles, setTextStyles] = useState({ color: "white", fontSize: "30px" })
  let componentRef = useRef()

  return (
    <div style={{ display: "flex", margin: "20px auto", justifyContent: "center" }}>
      <ComponentToPrint ref={(el) => (componentRef = el)} text={text} textStyles={textStyles} setText={setText} />
      <div className={styles.imageStyleOptions}>
        <p className={styles.sectionHeading}>Hoodie Styles</p>
        <p className={styles.inputContainer}>
          <label>Colors</label>
          <span style={{ background: "black" }} className={styles.colorSelector} onClick={() => setHoodieBackground("black")} />
          <span style={{ background: "teal" }} className={styles.colorSelector} onClick={() => setHoodieBackground("teal")} />
          <span
            style={{ background: "#0a1628" }}
            className={styles.colorSelector}
            onClick={() => setHoodieBackground("#0a1628")}
          />
          <span
            style={{ background: "#a38063" }}
            className={styles.colorSelector}
            onClick={() => setHoodieBackground("#a38063")}
          />
          <span
            style={{ background: "#4c4c4c" }}
            className={styles.colorSelector}
            onClick={() => setHoodieBackground("#4c4c4c")}
          />
          <span
            style={{ background: "#1f2f20" }}
            className={styles.colorSelector}
            onClick={() => setHoodieBackground("#1f2f20")}
          />
        </p>
        <p className={styles.inputContainer}>
          <label>Text</label>
          <input
            className={styles.textAdder}
            style={{ width: "100%" }}
            type='text'
            value={text}
            placeholder='Type something to add to the Hoodie...'
            onChange={(e) => setText(e.target.value)}
          />
        </p>
        <p className={styles.inputContainer}>
          <label>Font size</label>
          <CustomSlider
            value={textStyles.fontSize.substring(0, textStyles.fontSize.length - 2)}
            onChange={(e) =>
              setTextStyles((prev) => {
                return { ...prev, fontSize: e.target.value + "px" }
              })
            }
            min={0}
            max={30}
            labelValue={textStyles.fontSize}
          />
        </p>

        <ReactToPrint className={styles.button} trigger={() => <button>Submit design!</button>} content={() => componentRef} />
      </div>
    </div>
  )
}

const ComponentToPrint = forwardRef(({ text, textStyles, setText }, ref) => {
  const { hoodieBackground } = useContext(SelectedImagesContext)

  const { selectedImages } = useContext(SelectedImagesContext)

  return (
    <div style={{ display: "flex" }}>
      <div style={{ background: hoodieBackground }} className={`editContainer ${styles.editorContainer}`} ref={ref}>
        <div>
          {selectedImages?.map((img, index) => {
            return <MovableImage key={index} imageIndex={index} image={img.image} styleObj={img.styleObj} />
          })}
        </div>
        <MovableImage isImage={false} content={text} textStyles={textStyles} setText={setText} />
        <img src='./cutout3.png' className={styles.hoodiePattern} />
      </div>
    </div>
  )
})

function MovableImage({ image, imageIndex, styleObj, isImage = true, content = "", textStyles = {}, setText }) {
  const { setSelectedImages, hoodieBackground } = useContext(SelectedImagesContext)

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
    <div className={`iqbalImage ${styles.movableImageContainer}`}>
      <div id={uniq} className={styles.movableImage}>
        {isImage ? (
          <div
            style={{
              width: styleObj.width,
              height: styleObj.height,
              borderRadius: styleObj.borderRadius,
              backgroundColor: hoodieBackground,
              backgroundBlendMode: styleObj.backgroundBlendMode,
              backgroundImage: `url(${image})`,
              backgroundSize: "cover",
              backgroundPosition: `${styleObj.backgroundX}% ${styleObj.backgroundY}%`,
              boxShadow: `0 0 ${styleObj.fadeRadius}px ${styleObj.fadeIntensity}px ${hoodieBackground} inset`,
            }}
            onDoubleClick={() => doubleClickHandler(image, imageIndex)}
          />
        ) : (
          <div style={{ ...textStyles, cursor: "pointer" }} onDoubleClick={() => setText("")}>
            {content}
          </div>
        )}
      </div>

      <Moveable
        className={styles.movable_custom}
        target={`#${uniq}`}
        draggable={true}
        scalable={true}
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
