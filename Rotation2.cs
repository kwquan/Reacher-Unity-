using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Rotation2 : MonoBehaviour
{
    private GameObject joint2;
    private float xAngle = 0;
    private float yAngle = -0.1f;
    private float zAngle = 0;
    // Start is called before the first frame update
    void Start()
    {
        joint2 = GameObject.Find("Joint2");
    }

    // Update is called once per frame
    void Update()
    {
        joint2.transform.Rotate(xAngle, yAngle, zAngle);

    }
}
