import { useEffect, useRef, useState } from "react"
import styles from "./configurator.module.css"
import { max } from "lodash"
import CustomButton from "../buttons/index"

function GridPoint({ active, i, j, setGrid }) {
  function onClickHandler(i, j) {
    setGrid(max([i, j]))
  }

  return (
    <p
      className={`${i <= active && j <= active ? styles.active : styles.inactive} ${styles.gridPoint}`}
      onClick={() => onClickHandler(i, j)}
    ></p>
  )
}

function Grid({ setGridDim }) {
  const divRef = useRef(null)
  const [temp, setTemp] = useState([])
  const [cols, setCols] = useState(4)

  function setGridHelper(point) {
    setCols(point)
    setGridDim(point)
  }

  function makeGrid() {
    const count = 5
    const children = []
    for (let i = 1; i <= count; i++) {
      for (let j = 1; j <= count; j++) {
        children.push(<GridPoint active={cols} i={i} j={j} setGrid={setGridHelper} />)
      }
    }
    setTemp(children)
  }

  useEffect(() => {
    makeGrid()
  }, [cols])

  return (
    <div className={styles.gridParent} ref={divRef}>
      {temp.map((child, index) => (
        <span key={index}>{child}</span>
      ))}
    </div>
  )
}

export default function Configurator({ getImagesHandler, loading }) {
  const [numImages, setNumImages] = useState(5)
  const [gridDim, setGridDim] = useState(4)

  return (
    <div
      style={{
        marginLeft: 20,
        paddingLeft: 20,
        borderLeft: "1px solid #555",
        display: "flex",
        flexDirection: "column",
        justifyContent: "space-between",
      }}
    >
      <div>
        Configurator
        <Grid setGridDim={setGridDim} />
        <div className={styles.count_input}>
          <label>Number of Images</label>
          <input type='number' value={numImages} onChange={(e) => setNumImages(e.target.value)} />
        </div>
      </div>
      <CustomButton
        style={{ margin: "10px auto", alignSelf: "flex-end" }}
        onClick={() => getImagesHandler({ count: numImages, gridDim: gridDim })}
        loading={loading}
      >
        Get Images
      </CustomButton>
    </div>
  )
}
